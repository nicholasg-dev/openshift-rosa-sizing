#!/usr/bin/env python3
"""
OpenShift ROSA Sizing Tool - Metric and Sizing Collection Script
Version: 2.5

This script collects metrics from an OpenShift cluster's Prometheus instance
and the cluster's allocated node sizing for use in sizing ROSA clusters.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
import re # Import re for parsing resource strings

# urllib3 and requests will be imported after dependency check

# Default configuration
DEFAULT_OUTPUT = "cluster_data.json" # Renamed output file to be more general
DEFAULT_DAYS = 14 # Increased default days slightly for better history
DEFAULT_STEP = "1h"
DEFAULT_VERIFY_SSL = False

# Constants for Prometheus Queries
QUERY_CPU_USAGE = "sum(node_namespace_pod_container:container_cpu_usage_seconds_total:sum_irate)"
QUERY_MEMORY_USAGE = "sum(node_namespace_pod_container:container_memory_working_set_bytes)"
QUERY_POD_COUNT = "sum(kube_pod_info)"
QUERY_STORAGE_USAGE = "sum(kubelet_volume_stats_used_bytes)" # This is volume usage, not node disk capacity
# Added queries for resource requests and limits (peaks)
QUERY_CPU_REQUESTS_PEAK = "sum(kube_pod_container_resource_requests{resource='cpu'})"
QUERY_MEMORY_REQUESTS_PEAK = "sum(kube_pod_container_resource_requests{resource='memory'})"
QUERY_CPU_LIMITS_PEAK = "sum(kube_pod_container_resource_limits{resource='cpu'})"
QUERY_MEMORY_LIMITS_PEAK = "sum(kube_pod_container_resource_limits{resource='memory'})"


# Check dependencies and import them
def check_dependencies():
    print("Checking dependencies...")
    missing_deps = []
    # Check for basic Python version
    if sys.version_info < (3, 6):
        print("Error: This script requires Python 3.6 or later.")
        missing_deps.append("Python 3.6+")

    # Try to import required modules
    try:
        global urllib3
        import urllib3
        print("urllib3 found")
    except ImportError:
        missing_deps.append("urllib3")

    try:
        global requests
        import requests
        print("requests found")
    except ImportError:
        missing_deps.append("requests")

    if missing_deps:
        print("\nMissing requirements:")
        for dep in missing_deps:
            print(f"  - {dep}")
        if "Python 3.6+" not in missing_deps:
             print("\nPlease install the missing dependencies using one of these methods:")
             print("\n1. Using a virtual environment (recommended):")
             print("   python3 -m venv venv")
             print("   source venv/bin/activate")
             print("   pip install requests urllib3")
             print("\n2. User-specific installation:")
             print("   pip install --user requests urllib3")
             print("\n3. Direct package installation:")
             print(f"   pip install {' '.join(missing_deps)}")

        return False
    print("All dependencies found")
    return True

# Check dependencies first
if not check_dependencies():
    sys.exit(1)

# Now that we've imported urllib3, we can use it
try:
    # Disable SSL warnings
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except NameError:
    # If urllib3 import failed but we got here, something is wrong
    print("Error: urllib3 module not available despite dependency check.")
    sys.exit(1)

class PrometheusClient:
    """Client for interacting with Prometheus API in OpenShift"""
    # (Existing PrometheusClient class remains unchanged)
    def __init__(self, base_url, token, verify_ssl=False):
        """
        Initialize the Prometheus client

        Args:
            base_url (str): Base URL for Prometheus API
            token (str): Authentication token
            verify_ssl (bool): Whether to verify SSL certificates
        """
        print(f"Initializing Prometheus client with base URL: {base_url}")
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.verify_ssl = verify_ssl

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/json"
        })

        # Test connection
        try:
            print("Testing Prometheus connection...")
            response = self.session.get(
                f"{self.base_url}/api/v1/status/config",
                verify=self.verify_ssl
            )
            response.raise_for_status()
            print("Successfully connected to Prometheus")
        except requests.exceptions.SSLError:
            print("\nSSL verification failed. If this is expected, use --verify-ssl=false")
            raise
        except requests.exceptions.ConnectionError:
            print(f"\nFailed to connect to Prometheus at {self.base_url}")
            print("Please check if the URL is correct and accessible")
            raise
        except requests.exceptions.HTTPError as e:
            print(f"\nHTTP error connecting to Prometheus: {str(e)}")
            print("Please check your authentication token and permissions")
            raise
        except Exception as e:
            print(f"\nUnexpected error connecting to Prometheus: {str(e)}")
            raise

    def query_range(self, query, start, end, step):
        """
        Execute a range query against Prometheus

        Args:
            query (str): PromQL query
            start (int): Start timestamp
            end (int): End timestamp
            step (str): Step interval

        Returns:
            dict: Query results
        """
        endpoint = f"{self.base_url}/api/v1/query_range"
        params = {
            "query": query,
            "start": start,
            "end": end,
            "step": step
        }

        try:
            # print(f"Querying Prometheus: {query}") # Keep this less verbose during range queries
            response = self.session.get(
                endpoint,
                params=params,
                verify=self.verify_ssl
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "success":
                result = data.get("data", {}).get("result", [])
                # print(f"Query successful for {query[:50]}..., data points received: {len(result[0]['values']) if result else 0}")
                return data
            else:
                error = data.get("error", "Unknown error")
                print(f"Query failed for {query[:50]}...: {error}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Error querying Prometheus for {query[:50]}...: {str(e)}")
            return None
        except Exception as e:
            print(f"Unexpected error during query {query[:50]}...: {str(e)}")
            return None

    def query(self, query, timestamp=None):
        """
        Execute a single point query against Prometheus (useful for instant metrics)

        Args:
            query (str): PromQL query
            timestamp (int): Optional timestamp for the query

        Returns:
            dict: Query results
        """
        endpoint = f"{self.base_url}/api/v1/query"
        params = {
            "query": query
        }
        if timestamp:
            params["time"] = timestamp

        try:
            # print(f"Instant Querying Prometheus: {query}")
            response = self.session.get(
                endpoint,
                params=params,
                verify=self.verify_ssl
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "success":
                result = data.get("data", {}).get("result", [])
                # print(f"Instant Query successful for {query[:50]}..., data points received: {len(result)}")
                return data
            else:
                error = data.get("error", "Unknown error")
                print(f"Instant Query failed for {query[:50]}...: {error}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Error instant querying Prometheus for {query[:50]}...: {str(e)}")
            return None
        except Exception as e:
            print(f"Unexpected error during instant query {query[:50]}...: {str(e)}")
            return None


# Metric collection functions (Updated to use new keys and collect requests/limits)
def collect_metrics(client, start_time, end_time, step):
    """Collect all required metrics"""
    metrics_data = {}

    print("\nCollecting usage metrics (range queries)...")

    # CPU Usage
    print("  - CPU usage")
    cpu_data = client.query_range(QUERY_CPU_USAGE, start_time, end_time, step)
    if cpu_data and "data" in cpu_data and "result" in cpu_data["data"] and cpu_data["data"]["result"]:
        values = cpu_data["data"]["result"][0].get("values", [])
        timestamps = [entry[0] for entry in values]
        cpu_values = [float(entry[1]) for entry in values]
        metrics_data["cpu_usage"] = {
            "unit": "cores",
            "average": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
            "peak": max(cpu_values) if cpu_values else 0,
            # "values": [{"timestamp": ts, "value": val} for ts, val in zip(timestamps, cpu_values)] # Optional: save all values
        }
        print(f"    Collected {len(values)} CPU usage data points.")
    else:
        print("    Failed to collect CPU usage metrics or no data.")
        metrics_data["cpu_usage"] = {"unit": "cores", "average": 0, "peak": 0} # Ensure key exists

    # Memory Usage
    print("  - Memory usage")
    memory_data = client.query_range(QUERY_MEMORY_USAGE, start_time, end_time, step)
    if memory_data and "data" in memory_data and "result" in memory_data["data"] and memory_data["data"]["result"]:
        values = memory_data["data"]["result"][0].get("values", [])
        timestamps = [entry[0] for entry in values]
        memory_values = [float(entry[1]) / (1024**3) for entry in values]  # Convert bytes to GB
        metrics_data["memory_usage"] = {
            "unit": "GB",
            "average": sum(memory_values) / len(memory_values) if memory_values else 0,
            "peak": max(memory_values) if memory_values else 0,
             # "values": [{"timestamp": ts, "value": val} for ts, val in zip(timestamps, memory_values)] # Optional
        }
        print(f"    Collected {len(values)} Memory usage data points.")
    else:
         print("    Failed to collect Memory usage metrics or no data.")
         metrics_data["memory_usage"] = {"unit": "GB", "average": 0, "peak": 0} # Ensure key exists


    # Pod Count
    print("  - Pod count")
    pod_data = client.query_range(QUERY_POD_COUNT, start_time, end_time, step)
    if pod_data and "data" in pod_data and "result" in pod_data["data"] and pod_data["data"]["result"]:
        values = pod_data["data"]["result"][0].get("values", [])
        timestamps = [entry[0] for entry in values]
        pod_values = [int(float(entry[1])) for entry in values]
        metrics_data["pod_count"] = {
            "unit": "pods",
            "average": sum(pod_values) / len(pod_values) if pod_values else 0,
            "peak": max(pod_values) if pod_values else 0,
            # "values": [{"timestamp": ts, "value": val} for ts, val in zip(timestamps, pod_values)] # Optional
        }
        print(f"    Collected {len(values)} Pod count data points.")
    else:
         print("    Failed to collect Pod count metrics or no data.")
         metrics_data["pod_count"] = {"unit": "pods", "average": 0, "peak": 0} # Ensure key exists


    # Storage Usage (PV/PVC)
    print("  - Storage usage (PV/PVC)")
    storage_data = client.query_range(QUERY_STORAGE_USAGE, start_time, end_time, step)
    if storage_data and "data" in storage_data and "result" in storage_data["data"] and storage_data["data"]["result"]:
        values = storage_data["data"]["result"][0].get("values", [])
        timestamps = [entry[0] for entry in values]
        storage_values = [float(entry[1]) / (1024**3) for entry in values]  # Convert bytes to GB
        metrics_data["storage_usage_pvc"] = {
            "unit": "GB",
            "average": sum(storage_values) / len(storage_values) if storage_values else 0,
            "peak": max(storage_values) if storage_values else 0,
            # "values": [{"timestamp": ts, "value": val} for ts, val in zip(timestamps, storage_values)] # Optional
        }
        print(f"    Collected {len(values)} Storage usage data points.")
    else:
         print("    Failed to collect Storage usage metrics or no data.")
         metrics_data["storage_usage_pvc"] = {"unit": "GB", "average": 0, "peak": 0} # Ensure key exists


    print("\nCollecting request/limit peaks (instant queries at end time)...")

    # CPU Requests Peak (instant query)
    print("  - CPU requests peak")
    cpu_requests_data = client.query(QUERY_CPU_REQUESTS_PEAK, timestamp=end_time)
    if cpu_requests_data and "data" in cpu_requests_data and "result" in cpu_requests_data["data"] and cpu_requests_data["data"]["result"]:
        # Instant query returns a single value list
        metrics_data["cpu_usage"]["requests_peak"] = float(cpu_requests_data["data"]["result"][0].get("value", [end_time, 0])[1])
        print(f"    Collected CPU requests peak: {metrics_data['cpu_usage']['requests_peak']:.2f}")
    else:
        print("    Failed to collect CPU requests peak or no data.")
        metrics_data["cpu_usage"]["requests_peak"] = 0 # Ensure key exists


    # Memory Requests Peak (instant query)
    print("  - Memory requests peak")
    memory_requests_data = client.query(QUERY_MEMORY_REQUESTS_PEAK, timestamp=end_time)
    if memory_requests_data and "data" in memory_requests_data and "result" in memory_requests_data["data"] and memory_requests_data["data"]["result"]:
        metrics_data["memory_usage"]["requests_peak"] = float(memory_requests_data["data"]["result"][0].get("value", [end_time, 0])[1]) / (1024**3) # Convert bytes to GB
        print(f"    Collected Memory requests peak: {metrics_data['memory_usage']['requests_peak']:.2f} GB")
    else:
        print("    Failed to collect Memory requests peak or no data.")
        metrics_data["memory_usage"]["requests_peak"] = 0 # Ensure key exists

    # CPU Limits Peak (instant query)
    print("  - CPU limits peak")
    cpu_limits_data = client.query(QUERY_CPU_LIMITS_PEAK, timestamp=end_time)
    if cpu_limits_data and "data" in cpu_limits_data and "result" in cpu_limits_data["data"] and cpu_limits_data["data"]["result"]:
        metrics_data["cpu_usage"]["limits_peak"] = float(cpu_limits_data["data"]["result"][0].get("value", [end_time, 0])[1])
        print(f"    Collected CPU limits peak: {metrics_data['cpu_usage']['limits_peak']:.2f}")
    else:
        print("    Failed to collect CPU limits peak or no data.")
        metrics_data["cpu_usage"]["limits_peak"] = 0 # Ensure key exists

    # Memory Limits Peak (instant query)
    print("  - Memory limits peak")
    memory_limits_data = client.query(QUERY_MEMORY_LIMITS_PEAK, timestamp=end_time)
    if memory_limits_data and "data" in memory_limits_data and "result" in memory_limits_data["data"] and memory_limits_data["data"]["result"]:
        metrics_data["memory_usage"]["limits_peak"] = float(memory_limits_data["data"]["result"][0].get("value", [end_time, 0])[1]) / (1024**3) # Convert bytes to GB
        print(f"    Collected Memory limits peak: {metrics_data['memory_usage']['limits_peak']:.2f} GB")
    else:
        print("    Failed to collect Memory limits peak or no data.")
        metrics_data["memory_usage"]["limits_peak"] = 0 # Ensure key exists

    print("\nMetric collection complete.")

    return metrics_data


# Helper functions for parsing resource strings
def parse_cpu(cpu_string):
    """Parses CPU string (e.g., '2', '2500m') into cores (float)."""
    if isinstance(cpu_string, (int, float)):
        return float(cpu_string) # Assume it's already a number
    if not isinstance(cpu_string, str):
        print(f"Warning: Unexpected type for CPU string: {type(cpu_string)}. Attempting conversion.", file=sys.stderr)
        try:
            return float(cpu_string)
        except (ValueError, TypeError):
            print(f"Error: Could not parse invalid CPU string: '{cpu_string}'. Returning 0.", file=sys.stderr)
            return 0.0

    # Handle milli-cores
    if cpu_string.endswith('m'):
        try:
            return float(cpu_string[:-1]) / 1000.0
        except ValueError:
            print(f"Error: Could not parse milli-cores CPU string: '{cpu_string}'. Returning 0.", file=sys.stderr)
            return 0.0
    # Handle raw integer/float strings
    try:
        return float(cpu_string)
    except ValueError:
        print(f"Error: Could not parse raw CPU string: '{cpu_string}'. Returning 0.", file=sys.stderr)
        return 0.0


def parse_memory(memory_string):
    """Parses memory string (e.g., '1Gi', '16045056Ki') into GB (float)."""
    if isinstance(memory_string, (int, float)):
        # If it's a number, assume it's bytes and convert to GB
        return float(memory_string) / (1024**3)
    if not isinstance(memory_string, str):
         print(f"Warning: Unexpected type for Memory string: {type(memory_string)}. Attempting conversion from bytes.", file=sys.stderr)
         try:
             return float(memory_string) / (1024**3)
         except (ValueError, TypeError):
              print(f"Error: Could not parse invalid Memory string: '{memory_string}'. Returning 0.", file=sys.stderr)
              return 0.0


    units = {
        'Ei': 1024**6, 'E': 1000**6,
        'Pi': 1024**5, 'P': 1000**5,
        'Ti': 1024**4, 'T': 1000**4,
        'Gi': 1024**3, 'G': 1000**3,
        'Mi': 1024**2, 'M': 1000**2,
        'Ki': 1024**1, 'K': 1000**1,
    }
    match = re.match(r'(\d+)([A-Za-z]{1,2})', memory_string)
    if match:
        try:
            value_str, unit = match.groups()
            value = float(value_str)
            scale = units.get(unit, 1) # Default to 1 if unit not found (shouldn't happen with regex)
            return (value * scale) / (1024**3) # Convert to GB
        except ValueError:
             print(f"Error: Could not parse value in Memory string: '{memory_string}'. Returning 0.", file=sys.stderr)
             return 0.0
    # If no unit, assume raw bytes and convert to GB
    try:
        return float(memory_string) / (1024**3)
    except ValueError:
        print(f"Error: Could not parse raw Memory string: '{memory_string}'. Returning 0.", file=sys.stderr)
        return 0.0


def collect_cluster_sizing():
    """Collects cluster node sizing information using oc get nodes."""
    print("\nCollecting cluster node sizing via 'oc get nodes'...")
    try:
        result = subprocess.run(
            ['oc', 'get', 'nodes', '-o', 'json'],
            capture_output=True,
            text=True,
            check=True # Raise exception if command fails
        )
        nodes_data = json.loads(result.stdout)

        sizing_info = {
            "total_nodes": 0,
            "nodes_by_role": {},
            "nodes_by_instance_type": {},
            "total_capacity": {"cpu_cores": 0.0, "memory_gb": 0.0},
            "total_allocatable": {"cpu_cores": 0.0, "memory_gb": 0.0},
            "node_details": [] # Optional: Store details per node if needed
        }

        if not nodes_data.get('items'):
            print("No nodes found in the cluster via 'oc get nodes'. Sizing data will be empty.")
            return sizing_info # Return empty structure

        sizing_info["total_nodes"] = len(nodes_data['items'])
        worker_node_count = 0

        for node in nodes_data['items']:
            node_name = node['metadata']['name']
            roles = []
            # Extract roles from labels (e.g., node-role.kubernetes.io/worker)
            # The value is often just '', so check the label key prefix
            for label, value in node['metadata'].get('labels', {}).items():
                if label.startswith('node-role.kubernetes.io/'):
                     role = label.split('/')[-1]
                     roles.append(role)
                     sizing_info["nodes_by_role"][role] = sizing_info["nodes_by_role"].get(role, 0) + 1
                     if role == "worker":
                          worker_node_count += 1
            if not roles:
                 roles.append("unknown") # Node with no known role?

            # Extract instance type
            instance_type = node['metadata'].get('labels', {}).get('node.kubernetes.io/instance-type', 'unknown')
            sizing_info["nodes_by_instance_type"][instance_type] = sizing_info["nodes_by_instance_type"].get(instance_type, 0) + 1

            # Extract capacity and allocatable resources safely
            capacity = node.get('status', {}).get('capacity', {})
            allocatable = node.get('status', {}).get('allocatable', {})

            capacity_cpu_str = capacity.get('cpu', '0')
            capacity_memory_str = capacity.get('memory', '0')
            allocatable_cpu_str = allocatable.get('cpu', '0')
            allocatable_memory_str = allocatable.get('memory', '0')


            # Parse and sum total capacity (only sum worker node capacity for ROSA comparison)
            # ROSA control plane/infra capacity is fixed and not migrated from source
            # We are interested in the *worker* capacity of the source cluster
            if "worker" in roles:
                parsed_capacity_cpu = parse_cpu(capacity_cpu_str)
                parsed_capacity_memory = parse_memory(capacity_memory_str)
                sizing_info["total_capacity"]["cpu_cores"] += parsed_capacity_cpu
                sizing_info["total_capacity"]["memory_gb"] += parsed_capacity_memory

                # Parse and sum total allocatable (only sum worker node allocatable)
                parsed_allocatable_cpu = parse_cpu(allocatable_cpu_str)
                parsed_allocatable_memory = parse_memory(allocatable_memory_str)
                sizing_info["total_allocatable"]["cpu_cores"] += parsed_allocatable_cpu
                sizing_info["total_allocatable"]["memory_gb"] += parsed_allocatable_memory

            # Optional: Store individual node details
            sizing_info["node_details"].append({
                "name": node_name,
                "roles": roles,
                "instance_type": instance_type,
                "capacity": {
                    "cpu_cores": parse_cpu(capacity_cpu_str), # Store for all nodes
                    "memory_gb": parse_memory(capacity_memory_str) # Store for all nodes
                },
                "allocatable": {
                    "cpu_cores": parse_cpu(allocatable_cpu_str), # Store for all nodes
                    "memory_gb": parse_memory(allocatable_memory_str) # Store for all nodes
                }
            })

        # Add worker node count specifically
        sizing_info["worker_node_count"] = worker_node_count
        print(f"Cluster node sizing collected successfully. Found {worker_node_count} worker nodes.")
        return sizing_info

    except subprocess.CalledProcessError as e:
        print(f"\nError collecting cluster sizing: 'oc get nodes' command failed.")
        print(f"Stderr: {e.stderr.strip()}")
        print("Please ensure 'oc' is in your PATH and you are logged in with sufficient permissions to 'get nodes'.")
        return None
    except json.JSONDecodeError:
        print("\nError parsing 'oc get nodes' JSON output.")
        return None
    except KeyError as e:
        print(f"\nError processing node data structure: Missing key {e}")
        print("The output structure from 'oc get nodes' might be unexpected.")
        return None
    except Exception as e:
        print(f"\nUnexpected error collecting sizing: {str(e)}")
        return None


# (Existing oc related checks and setup functions remain unchanged)
def check_cluster():
    """Check if current cluster is OpenShift"""
    try:
        check_cmd = subprocess.run(
            ['oc', 'api-resources', '--api-group=apps.openshift.io'],
            capture_output=True, text=True
        )
        is_openshift = check_cmd.returncode == 0 and len(check_cmd.stdout.strip()) > 0
        if not is_openshift:
            print("\nWARNING: This does not appear to be an OpenShift cluster.")
            print("This tool is designed specifically for OpenShift clusters with Prometheus monitoring.")
            return False
        return True
    except FileNotFoundError:
        print("\nError: 'oc' command not found. Cannot check cluster type.")
        return False
    except Exception as e:
        print(f"\nError checking cluster type: {str(e)}")
        return False

def check_oc_login():
    """Check if user is logged into an OpenShift cluster"""
    try:
        # Check if oc command is available
        subprocess.run(['which', 'oc'], check=True, capture_output=True)

        # Get current login status
        status = subprocess.run(['oc', 'whoami'], capture_output=True, text=True)
        if status.returncode == 0:
            username = status.stdout.strip()

            # Get current cluster
            cluster = subprocess.run(
                ['oc', 'whoami', '--show-server'],
                capture_output=True, text=True
            )
            if cluster.returncode == 0:
                print(f"\nLogged in as: {username}")
                print(f"Current cluster: {cluster.stdout.strip()}")

                # Check if this is an OpenShift cluster
                if check_cluster():
                    # Ask for confirmation
                    confirm = input("\nDo you want to proceed with collecting data from this cluster? (y/n): ")
                    if confirm.lower() == 'y':
                        return True
            else:
                 print("\nCould not determine current cluster URL.")

        print("\nPlease log in to your OpenShift cluster first:")
        print("  oc login <cluster-url>")
        return False

    except subprocess.CalledProcessError:
        print("\nError: 'oc' command not found.")
        print("Please install the OpenShift CLI (oc) tool first.")
        return False
    except Exception as e:
        print(f"\nError checking login status: {str(e)}")
        return False

def get_current_cluster():
    """Get the current OpenShift cluster URL"""
    try:
        result = subprocess.run(['oc', 'whoami', '--show-server'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None

def detect_prometheus_url():
    """Auto-detect Prometheus URL from cluster"""
    print("\nAuto-detecting Prometheus URL...")
    try:
        # Try to get the route directly
        route_cmd = subprocess.run(
            ['oc', 'get', 'route', 'prometheus-k8s', '-n', 'openshift-monitoring', '-o', 'jsonpath={.spec.host}'],
            capture_output=True, text=True
        )

        if route_cmd.returncode == 0 and route_cmd.stdout.strip():
            prometheus_url = f"https://{route_cmd.stdout.strip()}"
            print(f"Detected Prometheus URL via Route: {prometheus_url}")
            return prometheus_url

        print("Prometheus Route not found. Attempting to construct URL based on cluster API...")
        # If route not found, try to construct from cluster URL
        cluster_url = get_current_cluster()
        if not cluster_url:
            print("Could not get current cluster URL to auto-detect Prometheus route.")
            return None

        # Extract cluster domain from API URL
        # Example: https://api.cluster-123.example.com:6443 -> apps.cluster-123.example.com
        if not cluster_url.startswith('https://api.'):
             print(f"Cluster API URL does not start with 'https://api.': {cluster_url}")
             print("Cannot reliably construct apps domain for Prometheus route.")
             return None

        # Remove 'https://api.' prefix and ':6443' suffix
        cluster_base = cluster_url[12:].split(':')[0]
        # Find the first dot after the cluster name (e.g., 'cluster-123')
        first_dot_after_cluster_name = cluster_base.find('.')
        if first_dot_after_cluster_name == -1:
             print(f"Could not parse cluster domain from API URL: {cluster_url}")
             return None

        # Take the part after the cluster name (e.g., 'example.com')
        domain_part = cluster_base[first_dot_after_cluster_name + 1:]
        apps_domain = f"apps.{domain_part}"

        # Construct Prometheus URL
        prometheus_url = f"https://prometheus-k8s-openshift-monitoring.{apps_domain}"
        print(f"Attempting Prometheus URL based on apps domain: {prometheus_url}")
        # Note: This constructed URL *might* not always work depending on DNS/network config,
        # but it's a common pattern. The route lookup is more reliable if it exists.
        return prometheus_url


    except Exception as e:
        print(f"Error detecting Prometheus URL: {str(e)}")
        return None

def setup_service_account():
    """Set up service account for Prometheus access and retrieve token"""
    print("\nSetting up service account for Prometheus access...")
    sa_name = "sizing-user"
    sa_namespace = "openshift-monitoring"

    try:
        # Step 1: Create/Update Service Account using oc apply
        sa_yaml = f"""apiVersion: v1
kind: ServiceAccount
metadata:
  name: {sa_name}
  namespace: {sa_namespace}"""

        print(f"Ensuring service account '{sa_name}' exists in namespace '{sa_namespace}'...")
        sa_cmd = subprocess.run(
            ['oc', 'apply', '-f', '-'],
            input=sa_yaml,
            text=True,
            capture_output=True
        )
        if sa_cmd.returncode != 0:
            print(f"Error ensuring service account exists: {sa_cmd.stderr.strip()}")
            # Don't necessarily exit here, might proceed with manual token
        else:
             print("Service account status checked.")


        # Step 2: Add cluster role binding
        print("Assigning cluster-monitoring-view role to service account...")
        role_binding_yaml = f"""apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: {sa_name}-monitoring-view
subjects:
- kind: ServiceAccount
  name: {sa_name}
  namespace: {sa_namespace}
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: cluster-monitoring-view"""

        role_cmd = subprocess.run(
            ['oc', 'apply', '-f', '-'],
            input=role_binding_yaml,
            text=True,
            capture_output=True
        )
        if role_cmd.returncode != 0:
            print(f"Error adding cluster role binding: {role_cmd.stderr.strip()}")
            print("You might need cluster-admin permissions to create ClusterRoleBindings.")
            print("Attempting to proceed, but token retrieval might fail if permissions are insufficient.")
            # Don't exit here, might proceed with manual token
        else:
             print("Role binding completed.")


        # Step 3: Get token - Try modern methods first
        print("Attempting to retrieve authentication token...")

        # Method 1: oc create token (OpenShift 4.11+)
        print("Trying 'oc create token' method (requires OpenShift 4.11+)...")
        token_cmd = subprocess.run(
            ['oc', 'create', 'token', sa_name, '-n', sa_namespace, '--duration=24h'], # Request a 24h token
            capture_output=True,
            text=True
        )
        if token_cmd.returncode == 0 and token_cmd.stdout.strip():
            print("Successfully retrieved token using 'oc create token'.")
            return token_cmd.stdout.strip()
        else:
            print(f" 'oc create token' failed: {token_cmd.stderr.strip()}")


        # Method 2: Create TokenRequest via API (pre-4.11 or if create token fails)
        print("Trying TokenRequest API method (pre-4.11)...")
        # Using a temporary file or stdin for TokenRequest yaml is safer than embedding directly in cmd string
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".yaml") as tmp_file:
            token_request_yaml = f"""apiVersion: authentication.k8s.io/v1
kind: TokenRequest
metadata:
  name: {sa_name}-token-request
  namespace: {sa_namespace}
spec:
  audiences:
    - https://kubernetes.default.svc
  expirationSeconds: 86400 # Token valid for 24 hours
  boundObjectRef: # Binding ensures token is only valid for this SA
    kind: ServiceAccount
    name: {sa_name}
    namespace: {sa_namespace}"""
            tmp_file.write(token_request_yaml)
            tmp_file_path = tmp_file.name

        token_cmd = subprocess.run(
            ['oc', 'create', '-f', tmp_file_path],
            capture_output=True,
            text=True
        )
        # Clean up temp file
        os.unlink(tmp_file_path)
        # Clean up the TokenRequest object we just created
        subprocess.run(['oc', 'delete', 'tokenrequest', f'{sa_name}-token-request', '-n', sa_namespace, '--ignore-not-found'], capture_output=True)


        try:
            token_data = json.loads(token_cmd.stdout)
            if 'status' in token_data and 'token' in token_data['status']:
                print("Successfully retrieved token using TokenRequest.")
                return token_data['status']['token']
            else:
                 print(f"TokenRequest API response missing token: {token_cmd.stdout.strip()}")
        except json.JSONDecodeError:
            print("Failed to parse TokenRequest API response.")
        except Exception as token_req_e:
             print(f"Error during TokenRequest API call: {str(token_req_e)}")


        # If automatic methods fail, provide manual instructions
        print("\nAutomatic token retrieval failed.")
        print("Please manually obtain a token for service account 'sizing-user' in namespace 'openshift-monitoring'.")
        print("Methods:")
        print(f"1. For OpenShift 4.11+: oc create token {sa_name} -n {sa_namespace} --duration=24h")
        print(f"2. For OpenShift < 4.11: oc serviceaccounts get-token {sa_name} -n {sa_namespace}") # Use get-token for older versions
        print("\nNote: The service account and cluster role binding are already set up if the previous steps succeeded.")

        token = input("\nEnter the token manually: ").strip()
        if token:
            return token
        else:
            print("No token provided manually.")
            return None

    except FileNotFoundError:
         print("\nError: 'oc' command not found. Cannot perform service account setup.")
         return None
    except Exception as e:
        print(f"Error during service account setup: {str(e)}")
        return None


def main():
    """Main function to collect metrics and sizing"""
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)

    try:
        parser = argparse.ArgumentParser(description="OpenShift ROSA Sizing Tool - Metric and Sizing Collection")

        # Optional arguments
        parser.add_argument("--prometheus-url", help="Prometheus URL (will be auto-detected if not provided)")
        parser.add_argument("--token", help="Authentication token (will be auto-generated if not provided)")
        parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output file (default: {DEFAULT_OUTPUT})")
        parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help=f"Days of historical data (default: {DEFAULT_DAYS})")
        parser.add_argument("--step", default=DEFAULT_STEP, help=f"Query interval (default: {DEFAULT_STEP})")
        parser.add_argument("--verify-ssl", type=lambda x: x.lower() == 'true', default=False, help="Verify SSL certificates (true/false)")

        args = parser.parse_args()
        print("Arguments parsed successfully.")

        # Check OpenShift login and confirm cluster
        print("\nChecking OpenShift login...")
        if not check_oc_login():
            print("Error: You are not logged into a confirmed OpenShift cluster.")
            sys.exit(1)

        # Initialize data structure
        collected_data = {
            "metadata": {
                "collection_timestamp": int(time.time()), # Use timestamp for easy processing
                "collection_date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
            "cluster_sizing": {}, # Initialize empty
            "metrics": {} # Initialize empty
        }

        # Collect cluster sizing information first (requires oc login)
        cluster_sizing = collect_cluster_sizing()
        if cluster_sizing:
            collected_data["cluster_sizing"] = cluster_sizing
        else:
            print("Warning: Failed to collect cluster sizing information.")


        # Get Prometheus URL
        prometheus_url = args.prometheus_url
        if not prometheus_url:
            print("\nAttempting to detect Prometheus URL...")
            prometheus_url = detect_prometheus_url()
            if not prometheus_url:
                print("Warning: Could not detect Prometheus URL. Skipping metric collection.")


        # Get authentication token
        token = args.token
        if not token and prometheus_url: # Only try setup SA if we have a Prometheus URL
            print("\nSetting up authentication for Prometheus...")
            token = setup_service_account()
            if not token:
                print("Warning: Could not obtain authentication token. Skipping metric collection.")


        # Create Prometheus client and collect metrics if URL and token are available
        client = None
        if prometheus_url and token:
            print("\nConnecting to Prometheus...")
            try:
                client = PrometheusClient(prometheus_url, token, args.verify_ssl)
            except Exception as e:
                print(f"\nFailed to connect to Prometheus. Metrics will not be collected: {str(e)}")
                client = None # Ensure client is None if connection fails
        elif prometheus_url and not token:
             print("\nAuthentication token not available. Skipping metric collection.")
        elif not prometheus_url:
             print("\nPrometheus URL not available. Skipping metric collection.")


        # Collect metrics if client is available
        if client:
            end_time = int(time.time())
            start_time = end_time - (args.days * 24 * 60 * 60)

            # Add data range info to metadata
            collected_data["metadata"]["prometheus_data_range"] = {
                "start_time_ts": start_time,
                "end_time_ts": end_time,
                "days": args.days,
                "step": args.step,
                 "start_date_iso": datetime.datetime.fromtimestamp(start_time, datetime.timezone.utc).isoformat(),
                 "end_date_iso": datetime.datetime.fromtimestamp(end_time, datetime.timezone.utc).isoformat(),
            }

            metrics_data = collect_metrics(client, start_time, end_time, args.step)
            if metrics_data:
                collected_data["metrics"] = metrics_data
        else:
            print("\nPrometheus client not available. Skipping metric collection.")
            # Add placeholder data range info even if metrics were skipped
            collected_data["metadata"]["prometheus_data_range"] = {
                "start_time_ts": None, "end_time_ts": None, "days": args.days,
                "step": args.step, "start_date_iso": None, "end_date_iso": None,
                "note": "Metric collection skipped or failed."
            }


        # Save collected data to file
        if os.path.exists(args.output):
            backup_file = f"{args.output}.bak"
            print(f"\nCreating backup of existing output file: {backup_file}")
            try:
                os.rename(args.output, backup_file)
            except OSError as e:
                 print(f"Warning: Could not create backup file {backup_file}: {str(e)}")


        with open(args.output, 'w') as f:
            json.dump(collected_data, f, indent=2)

        print(f"\nCollection complete. Data saved to {args.output}")

        # Print summary
        print("\n--- Collection Summary ---")
        print("Cluster Sizing:")
        if collected_data.get("cluster_sizing"):
             sizing = collected_data["cluster_sizing"]
             print(f"  Total Nodes: {sizing.get('total_nodes', 'N/A')}")
             print(f"  Worker Nodes: {sizing.get('worker_node_count', 'N/A')}")
             print(f"  Total Worker Capacity: CPU {sizing.get('total_capacity', {}).get('cpu_cores', 0.0):.2f} cores, Memory {sizing.get('total_capacity', {}).get('memory_gb', 0.0):.2f} GB")
             print(f"  Total Worker Allocatable: CPU {sizing.get('total_allocatable', {}).get('cpu_cores', 0.0):.2f} cores, Memory {sizing.get('total_allocatable', {}).get('memory_gb', 0.0):.2f} GB")
        else:
             print("  Failed to collect sizing data.")

        print("\nPrometheus Metrics (Peak Usage):")
        if collected_data.get("metrics"):
             metrics = collected_data["metrics"]
             cpu_usage = metrics.get("cpu_usage", {})
             mem_usage = metrics.get("memory_usage", {})
             pod_count = metrics.get("pod_count", {})
             storage_usage = metrics.get("storage_usage_pvc", {})
             print(f"  CPU Usage: {cpu_usage.get('peak', 0.0):.2f} cores peak, {cpu_usage.get('average', 0.0):.2f} cores average (Requests Peak: {cpu_usage.get('requests_peak', 0.0):.2f})")
             print(f"  Memory Usage: {mem_usage.get('peak', 0.0):.2f} GB peak, {mem_usage.get('average', 0.0):.2f} GB average (Requests Peak: {mem_usage.get('requests_peak', 0.0):.2f})")
             print(f"  Pod Count: {int(pod_count.get('peak', 0)):d} peak, {int(pod_count.get('average', 0)):d} average")
             print(f"  Storage Usage (PV/PVC): {storage_usage.get('peak', 0.0):.2f} GB peak, {storage_usage.get('average', 0.0):.2f} GB average")
             print(f"  Data collected over {collected_data['metadata']['prometheus_data_range'].get('days', 'N/A')} days with step {collected_data['metadata']['prometheus_data_range'].get('step', 'N/A')}")

        else:
             print("  Metric collection skipped or failed.")

        print("\nNext steps:")
        print(f"1. Review the collected data in the output file: {args.output}")
        print("2. Run calculate_sizing.py to generate sizing recommendations:")
        print(f"   python3 calculate_sizing.py --input {args.output} --redundancy 1.2") # Adjust redundancy as needed


    except Exception as e:
        print(f"\nAn unhandled error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()