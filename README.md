# OpenShift ROSA Sizing Tool

A comprehensive toolset for collecting metrics and calculating sizing recommendations for Red Hat OpenShift Service on AWS (ROSA) clusters.

## Features

### Metric Collection (collect_metrics.py)
- Collects key metrics from an OpenShift cluster's Prometheus instance
- Analyzes CPU usage, memory usage, pod count, and storage usage
- Calculates average and peak values for proper sizing
- Supports historical data analysis
- Includes cleanup and archival options

### Sizing Calculator (calculate_sizing.py)
- Analyzes collected metrics to provide ROSA sizing recommendations
- Suggests optimal AWS instance types based on workload patterns
- Calculates required node counts with redundancy
- Provides storage sizing recommendations
- Generates both JSON and human-readable reports

## Prerequisites

- Python 3.6+
- `requests` library
- Access to an OpenShift cluster with Prometheus
- A service account token with cluster-monitoring-view role
- AWS account for ROSA deployment

## Usage

### 1. Collect Metrics

First, collect metrics from your OpenShift cluster:

```bash
./collect_metrics.py --prometheus-url <URL> --token <TOKEN> [options]
```

#### Metric Collection Options

- `--output FILENAME` - Output file path (default: cluster_metrics.json)
- `--days DAYS` - Number of days of historical data to analyze (default: 7)
- `--step INTERVAL` - Query step interval (e.g., 1h, 30m, 5m) (default: 1h)
- `--verify-ssl` - Verify SSL certificates (default: False for self-signed certs)

#### Cleanup Options

- `--cleanup` - Clean up temporary files and reset environment
- `--remove-backups` - Remove backup files during cleanup
- `--remove-outputs` - Remove output files during cleanup
- `--logout` - Logout from OpenShift during cleanup
- `--archive` - Create a timestamped archive of output files

### 2. Calculate Sizing

Then, analyze the collected metrics to get sizing recommendations:

```bash
./calculate_sizing.py --input metrics.json [options]
```

#### Sizing Calculator Options

- `--input FILENAME` - Input metrics file (from collect_metrics.py)
- `--output FILENAME` - Output file for recommendations (default: rosa_sizing.json)
- `--format FORMAT` - Output format: json or text (default: json)
- `--redundancy FACTOR` - Redundancy factor (e.g., 1.3 = 30% extra capacity)

## Examples

### 1. Collecting Metrics

```bash
./collect_metrics.py \
  --prometheus-url https://prometheus-k8s-openshift-monitoring.apps.cluster.example.com \
  --token eyJhbGciOiJSUzI1NiIs... \
  --days 30 \
  --step 1h \
  --output my_cluster_metrics.json
```

### 2. Generating Sizing Recommendations

```bash
./calculate_sizing.py \
  --input my_cluster_metrics.json \
  --format text \
  --redundancy 1.3 \
  --output my_cluster_sizing.txt
```

## Output Files

### Metrics Collection Output (collect_metrics.py)
- JSON file containing:
  - CPU usage metrics (average, peak, minimum)
  - Memory usage metrics
  - Pod count statistics
  - Storage usage data
  - Node count information
  - Resource request metrics
  - Namespace statistics
  - Collection metadata

### Sizing Recommendations Output (calculate_sizing.py)
- JSON format:
  - Detailed sizing recommendations
  - Instance type suggestions
  - Node count calculations
  - Storage requirements
  - Utilization projections
- Text format:
  - Human-readable summary
  - Configuration details
  - Sizing rationale
  - Implementation notes

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
