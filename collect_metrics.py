#!/usr/bin/env python3
"""
OpenShift ROSA Sizing Tool - Metric Collection Script
Version: 2.0

This script collects metrics from an OpenShift cluster's Prometheus instance
for use in sizing ROSA clusters.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
import urllib3
import requests

# Default configuration
DEFAULT_OUTPUT = "cluster_metrics.json"
DEFAULT_DAYS = 7
DEFAULT_STEP = "1h"
DEFAULT_VERIFY_SSL = False

# Constants for Prometheus Queries
QUERY_CPU_USAGE = "sum(node_namespace_pod_container:container_cpu_usage_seconds_total:sum_irate)"
QUERY_MEMORY_USAGE = "sum(node_namespace_pod_container:container_memory_working_set_bytes)"
QUERY_POD_COUNT = "sum(kube_pod_info)"
QUERY_STORAGE_USAGE = "sum(kubelet_volume_stats_used_bytes)"

# Check dependencies before importing them
def check_dependencies():
    print("Checking dependencies...")
    missing_deps = []
    try:
        import urllib3
        print("urllib3 found")
    except ImportError:
        missing_deps.append("urllib3")
    
    try:
        import requests
        print("requests found")
    except ImportError:
        missing_deps.append("requests")
    
    if missing_deps:
        print("\nMissing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nPlease install the missing dependencies using one of these methods:")
        print("\n1. Using a virtual environment (recommended):")
        print("   python3 -m venv venv")
        print("   source venv/bin/activate")
        print("   pip install -r requirements.txt")
        print("\n2. User-specific installation:")
        print("   pip install --user -r requirements.txt")
        print("\n3. Direct package installation:")
        print(f"   pip install {' '.join(missing_deps)}")
        return False
    print("All dependencies found")
    return True

# Check dependencies first
if not check_dependencies():
    sys.exit(1)

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class PrometheusClient:
    """Client for interacting with Prometheus API in OpenShift"""
    
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
            print(f"Querying Prometheus: {query}")
            response = self.session.get(
                endpoint, 
                params=params, 
                verify=self.verify_ssl
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "success":
                result = data.get("data", {}).get("result", [])
                if result:
                    print("Query successful, data points received")
                    return data
                else:
                    print("Query successful but no data points found")
                    return None
            else:
                error = data.get("error", "Unknown error")
                print(f"Query failed: {error}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error querying Prometheus: {str(e)}")
            return None
        except Exception as e:
            print(f"Unexpected error during query: {str(e)}")
            return None

def collect_cpu_metrics(client, start_time, end_time, step):
    """Collect CPU usage metrics"""
    print("Collecting CPU metrics...")
    cpu_data = client.query_range(QUERY_CPU_USAGE, start_time, end_time, step)
    
    if not cpu_data or "data" not in cpu_data or "result" not in cpu_data["data"]:
        print("Failed to collect CPU metrics")
        return None
    
    # Process the results
    values = []
    if cpu_data["data"]["result"]:
        values = cpu_data["data"]["result"][0]["values"]
    
    # Extract timestamps and values
    timestamps = [entry[0] for entry in values]
    cpu_values = [float(entry[1]) for entry in values]
    
    # Calculate statistics
    avg_cpu = sum(cpu_values) / len(cpu_values) if cpu_values else 0
    max_cpu = max(cpu_values) if cpu_values else 0
    
    return {
        "unit": "cores",
        "average": avg_cpu,
        "peak": max_cpu,
        "values": [{
            "timestamp": ts,
            "value": val
        } for ts, val in zip(timestamps, cpu_values)]
    }

def collect_memory_metrics(client, start_time, end_time, step):
    """Collect memory usage metrics"""
    print("Collecting memory metrics...")
    memory_data = client.query_range(QUERY_MEMORY_USAGE, start_time, end_time, step)
    
    if not memory_data or "data" not in memory_data or "result" not in memory_data["data"]:
        print("Failed to collect memory metrics")
        return None
    
    # Process the results
    values = []
    if memory_data["data"]["result"]:
        values = memory_data["data"]["result"][0]["values"]
    
    # Extract timestamps and values (convert bytes to GB)
    timestamps = [entry[0] for entry in values]
    memory_values = [float(entry[1]) / (1024 * 1024 * 1024) for entry in values]  # Convert to GB
    
    # Calculate statistics
    avg_memory = sum(memory_values) / len(memory_values) if memory_values else 0
    max_memory = max(memory_values) if memory_values else 0
    
    return {
        "unit": "GB",
        "average": avg_memory,
        "peak": max_memory,
        "values": [{
            "timestamp": ts,
            "value": val
        } for ts, val in zip(timestamps, memory_values)]
    }

def collect_pod_metrics(client, start_time, end_time, step):
    """Collect pod count metrics"""
    print("Collecting pod count metrics...")
    pod_data = client.query_range(QUERY_POD_COUNT, start_time, end_time, step)
    
    if not pod_data or "data" not in pod_data or "result" not in pod_data["data"]:
        print("Failed to collect pod metrics")
        return None
    
    # Process the results
    values = []
    if pod_data["data"]["result"]:
        values = pod_data["data"]["result"][0]["values"]
    
    # Extract timestamps and values
    timestamps = [entry[0] for entry in values]
    pod_values = [int(float(entry[1])) for entry in values]
    
    # Calculate statistics
    avg_pods = sum(pod_values) / len(pod_values) if pod_values else 0
    max_pods = max(pod_values) if pod_values else 0
    
    return {
        "unit": "pods",
        "average": avg_pods,
        "peak": max_pods,
        "values": [{
            "timestamp": ts,
            "value": val
        } for ts, val in zip(timestamps, pod_values)]
    }

def collect_storage_metrics(client, start_time, end_time, step):
    """Collect storage usage metrics"""
    print("Collecting storage metrics...")
    storage_data = client.query_range(QUERY_STORAGE_USAGE, start_time, end_time, step)
    
    if not storage_data or "data" not in storage_data or "result" not in storage_data["data"]:
        print("Failed to collect storage metrics")
        return None
    
    # Process the results
    values = []
    if storage_data["data"]["result"]:
        values = storage_data["data"]["result"][0]["values"]
    
    # Extract timestamps and values (convert bytes to GB)
    timestamps = [entry[0] for entry in values]
    storage_values = [float(entry[1]) / (1024 * 1024 * 1024) for entry in values]  # Convert to GB
    
    # Calculate statistics
    avg_storage = sum(storage_values) / len(storage_values) if storage_values else 0
    max_storage = max(storage_values) if storage_values else 0
    
    return {
        "unit": "GB",
        "average": avg_storage,
        "peak": max_storage,
        "values": [{
            "timestamp": ts,
            "value": val
        } for ts, val in zip(timestamps, storage_values)]
    }

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
                    confirm = input("\nDo you want to proceed with collecting metrics from this cluster? (y/n): ")
                    if confirm.lower() == 'y':
                        return True
            
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
        
        if route_cmd.returncode == 0 and route_cmd.stdout:
            prometheus_url = f"https://{route_cmd.stdout.strip()}"
            print(f"Detected Prometheus URL: {prometheus_url}")
            return prometheus_url
            
        # If route not found, try to construct from cluster URL
        cluster_url = get_current_cluster()
        if not cluster_url:
            return None
            
        # Extract cluster domain from API URL
        # Example: https://api.cluster-123.example.com:6443 -> apps.cluster-123.example.com
        if not cluster_url.startswith('https://api.'):
            print("Error: Invalid cluster API URL format")
            return None
            
        # Remove 'https://api.' prefix and ':6443' suffix
        cluster_base = cluster_url[12:].split(':')[0]
        apps_domain = f"apps.{'.'.join(cluster_base.split('.')[1:])}"
        
        # Construct Prometheus URL
        prometheus_url = f"https://prometheus-k8s-openshift-monitoring.{apps_domain}"
        print(f"Detected Prometheus URL: {prometheus_url}")
        return prometheus_url
        
    except Exception as e:
        print(f"Error detecting Prometheus URL: {str(e)}")
        return None

def setup_service_account():
    """Set up service account for Prometheus access and retrieve token"""
    print("Setting up service account for Prometheus access...")
    sa_name = "sizing-user"
    sa_namespace = "openshift-monitoring"

    try:
        # Step 1: Create/Update Service Account using oc apply
        sa_yaml = f"""apiVersion: v1
kind: ServiceAccount
metadata:
  name: {sa_name}
  namespace: {sa_namespace}"""
        
        print(f"Ensuring service account '{sa_name}' exists...")
        sa_cmd = subprocess.run(
            ['oc', 'apply', '-f', '-'],
            input=sa_yaml,
            text=True,
            capture_output=True
        )
        if sa_cmd.returncode != 0:
            print(f"Error ensuring service account exists: {sa_cmd.stderr.strip()}")
            return None
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
            return None
        print("Role binding completed.")

        # Step 3: Get token - Try modern methods first
        print("Attempting to retrieve authentication token...")

        # Method 1: oc create token (OpenShift 4.11+)
        print("Trying 'oc create token' method...")
        token_cmd = subprocess.run(
            ['oc', 'create', 'token', sa_name, '-n', sa_namespace],
            capture_output=True,
            text=True
        )
        if token_cmd.returncode == 0 and token_cmd.stdout.strip():
            print("Successfully retrieved token using 'oc create token'.")
            return token_cmd.stdout.strip()

        # Method 2: Create TokenRequest via API
        print("Trying TokenRequest API method...")
        token_request_yaml = f"""apiVersion: authentication.k8s.io/v1
kind: TokenRequest
metadata:
  name: {sa_name}-token
  namespace: {sa_namespace}
spec:
  audiences:
    - https://kubernetes.default.svc
  expirationSeconds: 86400
  boundObjectRef:
    kind: ServiceAccount
    name: {sa_name}
    namespace: {sa_namespace}"""

        token_cmd = subprocess.run(
            ['oc', 'create', '-f', '-'],
            input=token_request_yaml,
            text=True,
            capture_output=True
        )
        try:
            token_data = json.loads(token_cmd.stdout)
            if 'status' in token_data and 'token' in token_data['status']:
                print("Successfully retrieved token using TokenRequest.")
                return token_data['status']['token']
        except json.JSONDecodeError:
            print("Failed to parse TokenRequest response.")

        # If automatic methods fail, provide manual instructions
        print("\nAutomatic token retrieval failed. Please use one of these methods:")
        print("\n1. For OpenShift 4.11 and newer:")
        print(f"   oc create token {sa_name} -n {sa_namespace}")
        print("\n2. For earlier versions:")
        print(f"   oc create -f - <<EOF\n{token_request_yaml}\nEOF")
        print("\nNote: The service account and role are already set up, you just need the token.")

        token = input("\nEnter the token: ").strip()
        if token:
            return token

    except Exception as e:
        print(f"Error in service account setup: {str(e)}")
        return None

def main():
    """Main function to collect metrics"""
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)

    try:
        parser = argparse.ArgumentParser(description="OpenShift ROSA Sizing Tool - Metric Collection")
        
        # Optional arguments
        parser.add_argument("--prometheus-url", help="Prometheus URL (will be auto-detected if not provided)")
        parser.add_argument("--token", help="Authentication token (will be auto-generated if not provided)")
        parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output file (default: {DEFAULT_OUTPUT})")
        parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help=f"Days of historical data (default: {DEFAULT_DAYS})")
        parser.add_argument("--step", default=DEFAULT_STEP, help=f"Query interval (default: {DEFAULT_STEP})")
        parser.add_argument("--verify-ssl", type=lambda x: x.lower() == 'true', default=False, help="Verify SSL certificates (true/false)")
        
        args = parser.parse_args()
        print("Arguments parsed successfully.")
        
        # Check OpenShift login
        print("\nChecking OpenShift login...")
        if not check_oc_login():
            print("Error: You are not logged into an OpenShift cluster.")
            
            sys.exit(1)
        
        # Get Prometheus URL
        prometheus_url = args.prometheus_url
        if not prometheus_url:
            print("\nAttempting to detect Prometheus URL...")
            prometheus_url = detect_prometheus_url()
            if not prometheus_url:
                print("Error: Could not detect Prometheus URL")
                sys.exit(1)
            print(f"Using Prometheus URL: {prometheus_url}")
        
        # Get authentication token
        token = args.token
        if not token:
            print("\nSetting up authentication...")
            token = setup_service_account()
            if not token:
                print("Error: Could not obtain authentication token")
                print("Please refer to the instructions above to obtain a token manually.")
                print("Then run this script with the --token argument.")
                sys.exit(1)
            print("Authentication token obtained successfully.")
        
        # Create Prometheus client
        print("\nConnecting to Prometheus...")
        client = PrometheusClient(prometheus_url, token, args.verify_ssl)
        
        # Calculate time range
        end_time = int(time.time())
        start_time = end_time - (args.days * 24 * 60 * 60)
        
        # Format timestamps for display
        start_date = datetime.datetime.fromtimestamp(start_time).isoformat()
        end_date = datetime.datetime.fromtimestamp(end_time).isoformat()
        print(f"\nCollecting metrics from {start_date} to {end_date}")
        print(f"Time period: {args.days} days with {args.step} intervals")
        
        # Initialize metrics structure
        metrics = {
            "metadata": {
                "collection_date": datetime.datetime.now().isoformat(),
                "start_date": start_date,
                "end_date": end_date,
                "days": args.days,
                "step": args.step
            },
            "metrics": {}
        }
        
        # Collect metrics with progress feedback
        print("\nCollecting metrics...")
        
        print("  - CPU metrics")
        cpu_data = collect_cpu_metrics(client, start_time, end_time, args.step)
        if cpu_data:
            metrics["metrics"]["cpu"] = cpu_data
        
        print("  - Memory metrics")
        memory_data = collect_memory_metrics(client, start_time, end_time, args.step)
        if memory_data:
            metrics["metrics"]["memory"] = memory_data
        
        print("  - Pod count metrics")
        pod_data = collect_pod_metrics(client, start_time, end_time, args.step)
        if pod_data:
            metrics["metrics"]["pods"] = pod_data
        
        print("  - Storage metrics")
        storage_data = collect_storage_metrics(client, start_time, end_time, args.step)
        if storage_data:
            metrics["metrics"]["storage"] = storage_data
        
        # Save metrics to file
        if os.path.exists(args.output):
            backup_file = f"{args.output}.bak"
            print(f"\nCreating backup of existing output file: {backup_file}")
            os.rename(args.output, backup_file)
        
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nMetrics saved to {args.output}")
        
        # Print summary with detailed error handling
        print("\nMetrics Summary:")

        # Check CPU metrics
        cpu_data = metrics["metrics"].get("cpu")
        if cpu_data and isinstance(cpu_data, dict) and cpu_data.get("peak") is not None and cpu_data.get("average") is not None:
            print(f"CPU Usage: Avg: {cpu_data['average']:.2f} cores, Peak: {cpu_data['peak']:.2f} cores")
        else:
            print("CPU Usage: Failed to collect or no data available.")

        # Check Memory metrics
        memory_data = metrics["metrics"].get("memory")
        if memory_data and isinstance(memory_data, dict) and memory_data.get("peak") is not None and memory_data.get("average") is not None:
            print(f"Memory Usage: Avg: {memory_data['average']:.2f} GB, Peak: {memory_data['peak']:.2f} GB")
        else:
            print("Memory Usage: Failed to collect or no data available.")

        # Check Pod metrics
        pod_data = metrics["metrics"].get("pods")
        if pod_data and isinstance(pod_data, dict) and pod_data.get("peak") is not None and pod_data.get("average") is not None:
            print(f"Pod Count: Avg: {pod_data['average']:.0f}, Peak: {pod_data['peak']:.0f}")
        else:
            print("Pod Count: Failed to collect or no data available.")

        # Check Storage metrics
        storage_data = metrics["metrics"].get("storage")
        if storage_data and isinstance(storage_data, dict) and storage_data.get("peak") is not None and storage_data.get("average") is not None:
            print(f"Storage Usage: Avg: {storage_data['average']:.2f} GB, Peak: {storage_data['peak']:.2f} GB")
        else:
            print("Storage Usage: Failed to collect or no data available.")
        
        print("\nNext steps:")
        print("1. Review the collected metrics in the output file")
        print("2. Run calculate_sizing.py to generate sizing recommendations:")
        print(f"   python3 calculate_sizing.py --input {args.output} --redundancy 1.3")

    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
