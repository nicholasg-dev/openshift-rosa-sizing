#!/usr/bin/env python3
# ============================================================================
# OpenShift Sizing - Metric Collection Script
# ============================================================================
#
# This script collects key metrics from an OpenShift cluster's Prometheus instance
# to help with sizing calculations for ROSA (Red Hat OpenShift Service on AWS).
#
# The script queries Prometheus for metrics such as CPU usage, memory usage,
# pod count, and storage usage over a specified time period. It then calculates
# average and peak values for each metric and saves the results to a JSON file.
#
# Usage:
#   ./collect_metrics.py --prometheus-url <URL> --token <TOKEN> [options]
#
# Options:
#   --output FILENAME    Output file path (default: cluster_metrics.json)
#   --days DAYS          Number of days of historical data to analyze (default: 7)
#   --step INTERVAL      Query step interval (e.g., 1h, 30m, 5m) (default: 1h)
#   --verify-ssl         Verify SSL certificates (default: False for self-signed certs)
#
# Cleanup Options:
#   --cleanup, --clean, --clean-up       Clean up temporary files and reset environment
#   --remove-backups, --remove-back-ups  Remove backup files during cleanup
#   --remove-outputs, --remove-files     Remove output files during cleanup
#   --logout, --log-out                  Logout from OpenShift during cleanup
#   --archive, --zip                     Create a timestamped archive of output files
#
# Requirements:
#   - Python 3.6+
#   - requests library
#   - Access to an OpenShift cluster with Prometheus
#   - A service account token with cluster-monitoring-view role
#
# Author: OpenShift Sizing Team
# ============================================================================

import requests
import json
import time
import sys
import os
import glob
import shutil
import argparse
import subprocess
from datetime import datetime, timedelta
import urllib3

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def check_dependencies():
    """
    Check if required dependencies are installed.
    
    Returns:
        bool: True if all dependencies are installed, False otherwise
    """
    # The requests module is already imported at the top level,
    # so if we got this far, it's available.
    # This function is here for future expansion to check other dependencies.
    return True

def check_openshift_login():
    """
    Check if the user is logged into an OpenShift cluster.
    
    This function runs the 'oc whoami' command to check if the user is logged in.
    It prints a warning if the user is not logged in, but does not exit the script.
    
    Returns:
        bool: True if logged in, False otherwise
    """
    try:
        result = subprocess.run(['oc', 'whoami'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               text=True, 
                               check=False)
        
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

def parse_arguments():
    """
    Parse command line arguments for the metric collection script.

    Returns:
        argparse.Namespace: The parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Collect metrics from Prometheus for OpenShift sizing.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--prometheus-url',
        required=True,
        help='Prometheus URL (e.g., https://prometheus-k8s-openshift-monitoring.apps.cluster.example.com)'
    )
    parser.add_argument(
        '--token',
        required=True,
        help='Bearer token for authentication to Prometheus'
    )

    # Optional arguments
    parser.add_argument(
        '--output',
        default='cluster_metrics.json',
        help='Output file path for the collected metrics'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days of historical data to analyze'
    )
    parser.add_argument(
        '--step',
        default='1h',
        help='Query step interval (e.g., 1h, 30m, 5m)'
    )
    parser.add_argument(
        '--verify-ssl',
        action='store_true',
        help='Verify SSL certificates (default: False for self-signed certs)'
    )

    # Cleanup options
    cleanup_group = parser.add_argument_group('Cleanup Options')
    cleanup_group.add_argument(
        '--cleanup', '--clean', '--clean-up',
        action='store_true',
        dest='cleanup',
        help='Clean up temporary files and reset environment after collecting metrics'
    )
    cleanup_group.add_argument(
        '--remove-backups', '--remove-backup', '--remove-back-ups',
        action='store_true',
        dest='remove_backups',
        help='Remove backup files during cleanup'
    )
    cleanup_group.add_argument(
        '--remove-outputs', '--remove-output', '--remove-files',
        action='store_true',
        dest='remove_outputs',
        help='Remove output files during cleanup (metrics file)'
    )
    cleanup_group.add_argument(
        '--logout', '--log-out',
        action='store_true',
        dest='logout',
        help='Logout from OpenShift during cleanup'
    )
    cleanup_group.add_argument(
        '--archive', '--create-archive', '--zip',
        action='store_true',
        dest='archive',
        help='Create a timestamped archive of output files before cleanup'
    )

    return parser.parse_args()

def query_prometheus(args, query, start_time, end_time):
    """
    Query Prometheus for metrics.

    Args:
        args (argparse.Namespace): Command line arguments
        query (str): Prometheus query string
        start_time (int): Start time in Unix timestamp
        end_time (int): End time in Unix timestamp

    Returns:
        dict: Prometheus query result
    """
    url = f"{args.prometheus_url}/api/v1/query_range"
    headers = {
        'Authorization': f'Bearer {args.token}',
        'Content-Type': 'application/json'
    }
    params = {
        'query': query,
        'start': start_time,
        'end': end_time,
        'step': args.step
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            params=params,
            verify=args.verify_ssl
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying Prometheus: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return None

def process_metric_data(data):
    """
    Process metric data from Prometheus.

    Args:
        data (dict): Prometheus query result

    Returns:
        dict: Processed metric data with average, peak, min, and samples
    """
    if not data or 'data' not in data or 'result' not in data['data']:
        return {
            'average': 0,
            'peak': 0,
            'min': 0,
            'samples': 0
        }

    results = data['data']['result']
    if not results:
        return {
            'average': 0,
            'peak': 0,
            'min': 0,
            'samples': 0
        }

    # Combine values from all result sets
    all_values = []
    for result in results:
        if 'values' in result:
            all_values.extend([float(v[1]) for v in result['values']])

    if not all_values:
        return {
            'average': 0,
            'peak': 0,
            'min': 0,
            'samples': 0
        }

    return {
        'average': sum(all_values) / len(all_values),
        'peak': max(all_values),
        'min': min(all_values),
        'samples': len(all_values)
    }

def collect_metrics(args):
    """
    Collect metrics from Prometheus.

    Args:
        args (argparse.Namespace): Command line arguments

    Returns:
        dict: Collected metrics
    """
    # Calculate time range
    end_time = int(time.time())
    start_time = end_time - (args.days * 24 * 60 * 60)

    print(f"\nCollecting metrics for the past {args.days} days...")
    print(f"Time range: {datetime.fromtimestamp(start_time)} to {datetime.fromtimestamp(end_time)}")
    print(f"Step interval: {args.step}")

    # Define metrics to collect
    metrics = {
        'cpu_usage': query_prometheus(
            args,
            'sum(node_namespace_pod_container:container_cpu_usage_seconds_total:sum_irate)',
            start_time,
            end_time
        ),
        'memory_usage': query_prometheus(
            args,
            'sum(container_memory_working_set_bytes{container!=""})/1024/1024/1024',
            start_time,
            end_time
        ),
        'pod_count': query_prometheus(
            args,
            'count(kube_pod_info)',
            start_time,
            end_time
        ),
        'pvc_storage': query_prometheus(
            args,
            'sum(kubelet_volume_stats_used_bytes)/1024/1024/1024',
            start_time,
            end_time
        ),
        'node_count': query_prometheus(
            args,
            'count(kube_node_info)',
            start_time,
            end_time
        ),
        'cpu_requests': query_prometheus(
            args,
            'sum(kube_pod_container_resource_requests{resource="cpu"})',
            start_time,
            end_time
        ),
        'memory_requests': query_prometheus(
            args,
            'sum(kube_pod_container_resource_requests{resource="memory"})/1024/1024/1024',
            start_time,
            end_time
        ),
        'namespace_count': query_prometheus(
            args,
            'count(kube_namespace_created)',
            start_time,
            end_time
        )
    }

    # Process metrics
    processed_metrics = {}
    for metric_name, metric_data in metrics.items():
        if metric_data:
            processed_metrics[metric_name] = process_metric_data(metric_data)
        else:
            print(f"Warning: No data returned for {metric_name}")
            processed_metrics[metric_name] = {
                'average': 0,
                'peak': 0,
                'min': 0,
                'samples': 0
            }

    # Add metadata
    processed_metrics['metadata'] = {
        'collection_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'days_collected': args.days,
        'step_interval': args.step,
        'prometheus_url': args.prometheus_url
    }

    # Save to file
    with open(args.output, 'w') as f:
        json.dump(processed_metrics, f, indent=2)

    print(f"\nMetrics saved to {args.output}")
    return processed_metrics

def load_cleanup_config():
    """
    Load the cleanup configuration from the config file.
    
    This function reads the cleanup_config.json file which specifies which files
    should be cleaned up, backed up, or archived. If the file doesn't exist,
    it uses default values.
    
    The configuration file has the following structure:
    {
        "temp_files": ["file1.txt", "*.tmp"],  # Files to remove during cleanup
        "output_files": {                     # Main output files
            "metrics": "cluster_metrics.json",
            "sizing": "rosa_sizing.json",
            "summary": "sizing_summary.txt"
        },
        "backup_files": ["*.bak"],           # Backup file patterns
        "archive_dir": "sizing_archives"     # Directory for archives
    }

    Returns:
        dict: The cleanup configuration
    """
    config_file = "cleanup_config.json"
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Default configuration if file doesn't exist
            return {
                "temp_files": ["prometheus_token.txt", "*.tmp", "*.log"],
                "output_files": {
                    "metrics": "cluster_metrics.json",
                    "sizing": "rosa_sizing.json",
                    "summary": "sizing_summary.txt"
                },
                "backup_files": ["*.bak"],
                "archive_dir": "sizing_archives"
            }
    except Exception as e:
        print(f"Warning: Error loading cleanup config: {e}")
        # Fallback to minimal configuration
        return {
            "temp_files": [],
            "output_files": {"metrics": "cluster_metrics.json"},
            "backup_files": ["*.bak"],
            "archive_dir": "sizing_archives"
        }

def create_archive(args, config):
    """
    Create a timestamped archive of output files.
    
    This function creates a ZIP archive containing the metrics file, sizing file,
    and summary file (if they exist). The archive is named with a timestamp and
    stored in the directory specified in the cleanup configuration.
    
    The archive is useful for preserving the results before cleaning up or for
    sharing the results with others.

    Args:
        args (argparse.Namespace): Command line arguments
        config (dict): Cleanup configuration with archive settings

    Returns:
        str: Path to the created archive, or None if archive creation failed
    """
    try:
        # Create archive directory if it doesn't exist
        archive_dir = config.get("archive_dir", "sizing_archives")
        if not os.path.exists(archive_dir):
            os.makedirs(archive_dir)
            
        # Create timestamp for archive name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"{archive_dir}/sizing_{timestamp}.zip"
        
        # Files to archive
        files_to_archive = []
        
        # Add metrics file if it exists
        metrics_file = args.output
        if os.path.exists(metrics_file):
            files_to_archive.append(metrics_file)
            
        # Add sizing file if it exists
        sizing_file = config["output_files"].get("sizing", "rosa_sizing.json")
        if os.path.exists(sizing_file):
            files_to_archive.append(sizing_file)
            
        # Add summary file if it exists
        summary_file = config["output_files"].get("summary", "sizing_summary.txt")
        if os.path.exists(summary_file):
            files_to_archive.append(summary_file)
            
        # Create the archive
        if files_to_archive:
            import zipfile
            with zipfile.ZipFile(archive_name, 'w') as zipf:
                for file in files_to_archive:
                    zipf.write(file, os.path.basename(file))
            print(f"\nCreated archive: {archive_name}")
            return archive_name
        else:
            print("\nNo files to archive.")
            return None
    except Exception as e:
        print(f"\nError creating archive: {e}")
        return None

def cleanup_environment(args):
    """Clean up after metric collection based on command line arguments."""
    print("\nPerforming cleanup...")
    
    # Load cleanup configuration
    config = load_cleanup_config()
    
    # Create archive if requested
    if args.archive:
        archive_path = create_archive(args, config)
        if archive_path:
            print(f"Created archive: {archive_path}")
    
    # Remove temporary files
    for pattern in config.get("temp_files", []):
        for file in glob.glob(pattern):
            try:
                os.remove(file)
                print(f"Removed temporary file: {file}")
            except Exception as e:
                print(f"Error removing {file}: {e}")
    
    # Remove backup files if requested
    if args.remove_backups:
        for pattern in config.get("backup_files", []):
            for file in glob.glob(pattern):
                try:
                    os.remove(file)
                    print(f"Removed backup file: {file}")
                except Exception as e:
                    print(f"Error removing {file}: {e}")
    
    # Remove output files if requested
    if args.remove_outputs:
        if os.path.exists(args.output):
            try:
                os.remove(args.output)
                print(f"Removed output file: {args.output}")
            except Exception as e:
                print(f"Error removing {args.output}: {e}")
    
    # Logout from OpenShift if requested
    if args.logout:
        try:
            subprocess.run(['oc', 'logout'], check=True)
            print("Logged out from OpenShift")
        except Exception as e:
            print(f"Error logging out: {e}")
    
    print("\nCleanup completed.")

def main():
    """
    Main function to run the metric collection process.

    This function performs the following steps:
    1. Checks dependencies to ensure required packages are installed
    2. Verifies OpenShift login status (warning only, doesn't exit)
    3. Parses command line arguments including cleanup options
    4. Collects metrics from Prometheus and saves to a JSON file
    5. Prints a summary of the collected metrics
    6. Performs cleanup tasks if any cleanup options are specified
    
    Cleanup options include:
    - --cleanup (or --clean, --clean-up): Basic cleanup
    - --remove-backups (or --remove-back-ups): Remove backup files
    - --remove-outputs (or --remove-files): Remove output files
    - --logout (or --log-out): Logout from OpenShift
    - --archive (or --zip): Create an archive of output files
    
    If any cleanup option is specified but --cleanup is not, cleanup is automatically enabled.
    """
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check OpenShift login status (warning only, doesn't exit)
    check_openshift_login()
    
    # Parse command line arguments
    args = parse_arguments()

    # Collect metrics from Prometheus
    metrics = collect_metrics(args)

    # Print a summary of the collected metrics
    print("\nMetrics Summary:")
    for metric, values in metrics.items():
        if metric != "metadata":
            print(f"{metric}:")
            print(f"  Average: {values['average']:.2f}")
            print(f"  Peak: {values['peak']:.2f}")
            print(f"  Min: {values['min']:.2f}")
            print(f"  Samples: {values['samples']}")

    print("\nThese metrics will be used for ROSA cluster sizing calculations.")
    print("Run the calculate_sizing.py script next to generate sizing recommendations.")
    
    # Perform cleanup if requested
    if args.cleanup or args.remove_backups or args.remove_outputs or args.logout or args.archive:
        # If any cleanup option is specified but --cleanup is not, enable it
        if not args.cleanup:
            args.cleanup = True
        cleanup_environment(args)

if __name__ == "__main__":
    main()
