# OpenShift ROSA Sizing Tool

A tool for collecting metrics and calculating sizing recommendations for Red Hat OpenShift Service on AWS (ROSA) clusters.

## Features

- Collects key metrics from an OpenShift cluster's Prometheus instance
- Analyzes CPU usage, memory usage, pod count, and storage usage
- Calculates average and peak values for proper sizing
- Supports historical data analysis
- Includes cleanup and archival options

## Prerequisites

- Python 3.6+
- `requests` library
- Access to an OpenShift cluster with Prometheus
- A service account token with cluster-monitoring-view role

## Usage

```bash
./collect_metrics.py --prometheus-url <URL> --token <TOKEN> [options]
```

### Options

- `--output FILENAME` - Output file path (default: cluster_metrics.json)
- `--days DAYS` - Number of days of historical data to analyze (default: 7)
- `--step INTERVAL` - Query step interval (e.g., 1h, 30m, 5m) (default: 1h)
- `--verify-ssl` - Verify SSL certificates (default: False for self-signed certs)

### Cleanup Options

- `--cleanup` - Clean up temporary files and reset environment
- `--remove-backups` - Remove backup files during cleanup
- `--remove-outputs` - Remove output files during cleanup
- `--logout` - Logout from OpenShift during cleanup
- `--archive` - Create a timestamped archive of output files

## Example

```bash
./collect_metrics.py \
  --prometheus-url https://prometheus-k8s-openshift-monitoring.apps.cluster.example.com \
  --token eyJhbGciOiJSUzI1NiIs... \
  --days 30 \
  --step 1h \
  --output my_cluster_metrics.json
```

## Output

The script generates a JSON file containing:
- CPU usage metrics (average, peak, minimum)
- Memory usage metrics
- Pod count statistics
- Storage usage data
- Node count information
- Resource request metrics
- Namespace statistics
- Collection metadata

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
