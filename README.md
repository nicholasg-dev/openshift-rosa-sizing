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

### 1. Python Environment
- Python 3.8+ recommended
- Use a virtual environment (venv) for dependency management
- Install dependencies using `pip install -r requirements.txt`

### 2. OpenShift Cluster Access
- Access to an OpenShift cluster with Prometheus monitoring enabled
- A service account with `cluster-monitoring-view` role
  - Required to access Prometheus metrics
  - See OpenShift documentation for creating service accounts and roles

### 3. AWS Account (for ROSA deployment)
- Needed for actual ROSA cluster deployment
- Not required for running the sizing tool itself

## Usage

### 1. Collect Metrics

First, collect metrics from your OpenShift cluster:

```bash
python collect_metrics.py --prometheus-url <URL> --token <TOKEN> [options]
```

#### Metric Collection Options

- `--output FILENAME` - Output file path (default: cluster_metrics.json)
- `--days DAYS` - Number of days of historical data (default: 7)
- `--step INTERVAL` - Query interval (e.g., 1h, 30m) (default: 1h)
- `--verify-ssl` - Verify SSL certificates (default: False)

#### Cleanup Options

- `--cleanup` - Basic cleanup
- `--remove-backups` - Remove backup files
- `--remove-outputs` - Remove output files
- `--logout` - Logout from OpenShift
- `--archive` - Create output archive

### 2. Calculate Sizing

Then, analyze collected metrics:

```bash
python calculate_sizing.py --input metrics.json [options]
```

#### Sizing Calculator Options

- `--input FILENAME` - Input metrics file
- `--output FILENAME` - Output file (default: rosa_sizing.json)
- `--format FORMAT` - Output format: json/text (default: json)
- `--redundancy FACTOR` - Redundancy factor (default: 1.3)

## Output Files

### Metrics Collection Output
- JSON file with:
  - CPU, memory, pod count, storage metrics
  - Collection metadata

### Sizing Recommendations
- JSON/Text output with:
  - Instance type suggestions
  - Node count recommendations
  - Storage requirements
  - Utilization projections
  - Rationale for recommendations

## ROSA Specific Considerations

### Cluster Components
- ROSA includes managed control plane and infra nodes
- Sizing recommendations focus on worker nodes

### Minimum Requirements
- Minimum 2-3 worker nodes recommended
- Multiple availability zones recommended for HA

### Instance Types
- Recommendations based on ROSA-supported AWS instance types
- Common families: m (general), c (compute), r (memory)

## Contributing

Contributions welcome! Submit Pull Requests.

## License

MIT License - see LICENSE file.
