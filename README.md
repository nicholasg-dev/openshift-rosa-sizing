# OpenShift ROSA Sizing Tool (v2.0)

A production-ready toolset for collecting real cluster metrics and generating accurate sizing recommendations for Red Hat OpenShift Service on AWS (ROSA) clusters.

## Key Improvements in v2.0

- **Real Data Only**: Removed all sample data dependencies
- **Enhanced Documentation**: Complete code comments and usage examples
- **ROSA-Specific Logic**: Added managed control plane considerations
- **Cost Estimation**: Integrated rough cost projections
- **Workload Profiling**: Automatic CPU/Memory-bound detection
- **Small Workload Support**: Improved handling of small workloads with 3-node minimum HA requirement

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

## Getting Started (Real Data Workflow)

1. **Ensure OpenShift CLI Access**:

   ```bash
   # Log in to your OpenShift cluster if not already logged in
   oc login --server=https://api.your-cluster.example.com:6443
   ```

   The tool will automatically verify your login status.

2. **Collect Metrics** (7-30 days recommended):

   ```bash
   python collect_metrics.py --days 14 --step 1h --output production_metrics.json
   ```

   The tool will automatically:

   - Detect your OpenShift cluster
   - Create a service account with appropriate permissions if needed
   - Generate and use the authentication token
   - Detect the Prometheus URL
   - Guide you through the process with interactive prompts

3. **Generate Recommendations**:

   ```bash
   python calculate_sizing.py --input production_metrics.json --output sizing_recommendations.json --redundancy 1.3
   ```

4. **Review Results**:

   ```bash
   # For JSON output
   cat sizing_recommendations.json

   # For text output (if you used --format text)
   cat sizing_recommendations.txt
   ```

## Prerequisites (Detailed)

### 1. Python Environment

- Python 3.8+ recommended
- The tool automatically checks for required dependencies (urllib3, requests) at runtime
- If dependencies are missing, the tool will provide clear installation instructions with three methods:
  - Using a virtual environment (recommended): `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
  - Using user-specific installation: `pip install --user -r requirements.txt`
  - Installing specific packages directly: `pip install urllib3 requests`

### 2. OpenShift Cluster Access

- Access to an OpenShift cluster with Prometheus monitoring enabled
- The tool will automatically:
  - Verify your OpenShift login status
  - Create a service account with the required `cluster-monitoring-view` role if needed
  - Generate and use the authentication token
  - Detect the Prometheus URL from your cluster

### 3. AWS Account (for ROSA deployment)

- Needed for actual ROSA cluster deployment
- Not required for running the sizing tool itself

## Usage

### 1. Collect Metrics

First, collect metrics from your OpenShift cluster:

```bash
python collect_metrics.py [options]
```

With the enhanced version, you no longer need to manually specify the Prometheus URL or token, as these are automatically detected and generated.

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

- ROSA includes managed control plane and infra nodes (typically on m5.xlarge instances)
- Sizing recommendations focus on worker nodes which handle application workloads

### Minimum Requirements

- Minimum 2 worker nodes recommended for basic functionality
- 3+ worker nodes recommended for high availability
- Multiple availability zones recommended for production environments

### Instance Types

- Recommendations based on all available x86_64 AWS instance types (including bare metal)
- Instance type families:
  - General Purpose (m5, m6i, etc.) - balanced CPU and memory
  - Compute Optimized (c5, c6i, etc.) - high CPU performance
  - Memory Optimized (r5, r6i, etc.) - high memory capacity
  - Storage Optimized (i3, i4i, etc.) - high storage performance
  - Bare Metal (metal instances) - dedicated hardware for high-performance requirements

### Sizing Algorithm

- Analyzes CPU and memory usage patterns from collected metrics
- Simulates different instance types and counts to find optimal configuration
- Considers both average and peak usage with redundancy factor
- Provides rationale for recommended configurations
- Applies utilization filtering to ensure efficient resource usage:
  - For small workloads with 3-node minimum HA requirement, no lower utilization bound is enforced
  - For larger deployments, utilization targets are 30-80% for both CPU and memory
  - Clearly identifies when the 3-node minimum HA requirement is the limiting factor

## Cost Estimation

While the tool doesn't directly pull AWS pricing data, you can use the instance type recommendations with AWS Pricing Calculator to estimate costs.

## Output & Reporting

### JSON Output

- Detailed sizing recommendations
- Intermediate calculations
- Rationale for recommendations

### Text Report

- Human-readable summary
- Configuration details
- Sizing rationale
- Detailed notes section including:
  - Information about the 3-node minimum HA requirement
  - Explanation of utilization targets for different deployment sizes
  - Guidance on interpreting low utilization for small workloads

## Contributing

Contributions welcome! Please submit Pull Requests with:

- Clear description of changes
- Any relevant test cases
- Documentation updates

## License

MIT License - see LICENSE file for details.
