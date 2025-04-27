# OpenShift ROSA Sizing Tool (v2.3)

A production-ready toolset for collecting real cluster metrics and generating accurate sizing recommendations for Red Hat OpenShift Service on AWS (ROSA) clusters.

**Important Note:** This tool sizes a target ROSA cluster based on the **workload demand (usage and requests)** observed in your existing OpenShift cluster metrics. It typically does **not** know or report the *total available capacity* of the original source cluster, as this information is not usually present in standard Prometheus metrics exports. The comparison provided is between the observed *workload demand* and the *recommended ROSA capacity* needed to handle that demand plus redundancy.

## Key Improvements in v2.3

- **Enhanced Output Structure**: Clear separation of observed workload metrics, calculated requirements, and recommended ROSA configuration summary in the JSON output.
- **Direct Workload-to-Capacity Comparison**: Text reports now include a dedicated "Sizing Comparison" section that contrasts observed workload peaks/requests with the calculated requirements and the total capacity of the recommended ROSA cluster configuration.
- **Improved Instance Coverage**: Added support for Graviton (ARM) instance types (m7g, c7g, r7g) and ensured bare metal options (like .metal) are included in the available instance types for recommendations.
- **Refined Sizing Logic**: Updates to workload profiling, node calculation basis explanations, and efficiency scoring to provide more robust and understandable recommendations.
- **Clearer Terminology**: Output uses clearer terms like "Observed Workload Metrics" vs. "Calculated ROSA Requirements" and "Recommended ROSA Configuration" to differentiate between the source workload and the target cluster size.
- **Improved Error Handling**: More robust handling of potentially missing data in input metrics.

### Includes improvements from v2.0 and v2.1

Like real data only, enhanced docs, ROSA-specific logic, automatic detection, workload profiling, small workload support, and utilization handling.

## Features

### Metric Collection (collect_metrics.py)

- Connects to an OpenShift cluster's Prometheus instance to collect key metrics (CPU usage/requests/limits, memory usage/requests/limits, pod count, storage usage).
- Analyzes historical data over a specified period (e.g., 7 or 30 days).
- Calculates peak and average values to understand workload characteristics and high watermarks.
- Automatically handles OpenShift authentication (service account, token).
- Detects Prometheus URL automatically by first checking the OpenShift route and then constructing from the cluster URL if needed.
- Saves metrics to a structured JSON file.

### Sizing Calculator (calculate_sizing.py)

- Reads the collected workload metrics from the JSON input file.
- Analyzes the workload profile (CPU-bound, Memory-bound, or Balanced).
- Calculates the total required resources (CPU, memory, pods, storage) for the target ROSA cluster based on observed peaks/requests plus a configurable redundancy factor.
- Evaluates a comprehensive list of relevant AWS instance types suitable for ROSA worker nodes (including General Purpose, Compute Optimized, Memory Optimized, Intel/AMD/Graviton, and bare metal).
- Determines the minimum number of nodes of each instance type required to meet the calculated resource needs and the minimum high availability requirement (3 nodes).
- Ranks potential worker node configurations using an efficiency score that considers factors like instance generation, resource utilization, network capability, and total node count.
- Generates detailed recommendations in both JSON and human-readable text formats.
- **Provides a comparison in the text output showing the observed workload demand, the calculated required resources, and the total capacity of the recommended ROSA cluster configuration.**

## Getting Started (Real Data Workflow)

1. Ensure OpenShift CLI Access:

```bash
# Log in to your OpenShift cluster if not already logged in
oc login --server=https://api.your-cluster.example.com:6443 --token=YOUR_TOKEN # Or other login methods
```

The `collect_metrics.py` tool requires `oc` access and will verify your login status and automatically create necessary resources (like a service account with `cluster-monitoring-view` role) if needed.

2. Collect Metrics (7-30 days recommended for capturing peak periods):

```bash
python collect_metrics.py --days 14 --step 1h --output production_metrics.json
```

The tool will automatically:

- Detect your OpenShift cluster details.
- Create a service account with appropriate permissions if needed.
- Generate and use an authentication token.
- Detect the Prometheus URL.
- Collect and process the required metrics.
- Save the output to the specified JSON file.

3. Generate Recommendations:

```bash
python calculate_sizing.py --input production_metrics.json --output sizing_recommendations.json --redundancy 1.3 --format text
```

Using `--format text` (as shown above) is recommended for initial review, as it includes the human-readable comparison table. You can also output to JSON.

4. Review Results:

```bash
# For JSON output (if you used --format json)
cat sizing_recommendations.json | less # Use less for large files

# For text output (if you used --format text, e.g., sizing_recommendations.txt)
cat sizing_recommendations.txt
```

**In the text report, look for the "OBSERVED WORKLOAD METRICS" section and the "SIZING COMPARISON" table** to understand the workload demand and how it compares to the recommended ROSA cluster capacity.

## Prerequisites (Detailed)

### 1. Python Environment

- Python 3.8+ recommended.
- The tool automatically checks for required dependencies (`requests`, `urllib3`) at runtime.
- If dependencies are missing, the tool will provide clear installation instructions. Using a virtual environment is the recommended approach:

```bash
python3 -m venv venv
source venv/bin/activate # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

*(Create a `requirements.txt` file containing `requests` and `urllib3` if you don't have one).*

### 2. OpenShift Cluster Access

- Access to an OpenShift cluster (version 4.x+) with Prometheus monitoring enabled and accessible from where you run the tool.
- The tool requires `oc` command-line access and appropriate permissions (automatic service account creation with `cluster-monitoring-view` role is attempted).

### 3. AWS Account (for ROSA deployment)

- An AWS account is required for *deploying* a ROSA cluster based on the recommendations.
- It is **not** required for simply running the sizing tool scripts (`collect_metrics.py` or `calculate_sizing.py`).

## ROSA Sizing Methodology & Logic

The `calculate_sizing.py` script employs the following methodology:

1.  **Observed Workload Demand**: Reads peak usage and peak requests for CPU, Memory, Pods, and Storage from the `collect_metrics.py` output. This represents the historical high-water mark of your workload's resource needs and resource reservations in the source cluster.
2.  **Sizing Baseline**: For CPU and Memory, the larger of the peak usage and peak requests is taken as the baseline resource need. For Pods and Storage, the peak usage/count is used as the baseline.
3.  **Redundancy Factor**: The sizing baseline is multiplied by a configurable redundancy factor (default 1.3, representing 30% overhead) to account for potential future growth, transient spikes, rolling updates, and node failures. This results in the "Calculated Required Resources" for the target ROSA cluster.
4.  **High Availability Minimum**: ROSA requires a minimum of 3 worker nodes for a highly available production cluster. The node count calculation will always recommend at least this minimum, even if the calculated required resources suggest fewer nodes would be sufficient.
5.  **Pod Density Limit**: A default limit of 110 pods per worker node is used as a conservative baseline, based on standard Kubernetes/OpenShift configurations and typical AWS/CNI limitations. The node count needed to support the required pod count is factored into the total node calculation.
6.  **Instance Evaluation**: The script evaluates a built-in list of relevant AWS instance types suitable for ROSA worker nodes. This list includes various families (General Purpose, Compute Optimized, Memory Optimized) and processor architectures (Intel, Graviton/ARM), including bare metal variants.
7.  **Node Count Calculation per Instance Type**: For each instance type, the script calculates how many nodes of that type would be needed to meet the "Calculated Required Resources" (CPU, Memory, Pods). The highest of these individual requirements, capped by the High Availability minimum, determines the total "Nodes Needed" for that specific instance type.
8.  **Efficiency Scoring**: Each potential configuration (instance type + nodes needed) is assigned an efficiency score. This score balances:
    - **Resource Utilization**: How closely the total capacity of the recommended nodes matches the "Calculated Required Resources" (aiming near a target utilization, e.g., 65%). Deviation is penalized.
    - **Instance Generation**: Newer generations typically offer better price/performance and features.
    - **Network Capability**: Higher network bandwidth is generally preferred for cluster communication and application traffic.
    - **Node Count**: Fewer nodes (closer to the HA minimum) are generally preferred for simplicity and potentially lower management overhead (within the constraints of meeting resource needs).
9.  **Ranking and Recommendation**: Configurations are sorted by their efficiency score (highest first) and the top options are presented. The absolute top option is highlighted as the "Recommended ROSA Configuration Summary".

## Interpreting the Output

The `calculate_sizing.py` script generates output in either JSON or text format.

### JSON Output Structure

The JSON output provides a detailed breakdown suitable for programmatic consumption. Key sections include:

- `report_metadata`: Information about the report generation.
- `observed_workload_metrics`: Peak and average usage and requests from your source cluster during the collection period. **This represents the historical workload DEMAND.**
- `calculated_rosa_requirements`: The resource, pod, and storage requirements calculated for the target ROSA cluster after considering the observed metrics (demand baseline) and applying the redundancy factor. This is the target size the recommendations aim to meet.
- `recommended_rosa_configuration_summary`: A summary of the top-ranked worker node configuration (instance type and node count), including the *total capacity* this configuration provides. **This represents the proposed ROSA cluster SUPPLY.**
- `worker_node_options`: A ranked list of other potential worker node configurations, detailing their capacity, utilization at the recommended node count, and efficiency score.
- `general_notes`: Important contextual information and considerations for ROSA sizing.

### Text Report Format (`--format text`)

The text report is designed for easy human readability and includes key sections:

- **SUMMARY**: A brief overview of the workload profile and the top-ranked ROSA worker node recommendation.
- **REPORT METADATA**: Information about the report and metrics collection period.
- **OBSERVED WORKLOAD METRICS (from source cluster)**: Details the peak and average usage and requests observed. **Crucially notes that this is the DEMAND, not the original cluster's capacity.**
- **CALCULATED ROSA REQUIREMENTS (Sizing Basis)**: Explains how the required resources for the target ROSA cluster were calculated from the observed demand and redundancy factor.
- **SIZING COMPARISON**: **This table provides a direct side-by-side comparison** of the Peak Observed Workload (Demand), the Calculated Required Resources (with redundancy), and the total Recommended ROSA Capacity (Supply) for CPU, Memory, Pods, and Storage. This is where you can directly see how the recommended ROSA size compares to the workload you observed.
- **WORKER NODE OPTIONS**: A ranked list of alternative instance type configurations, showing nodes needed, utilization, total capacity, and efficiency score for each.
- **GENERAL NOTES**: Important notes regarding ROSA architecture, HA requirements, storage, node pools, cost, and the nature of the recommendations as estimates.

## ROSA Specific Considerations (Detailed)

### Cluster Components

- A ROSA cluster includes Red Hat managed control plane nodes (typically 3 nodes) and infrastructure nodes (typically 3 nodes) running on smaller, managed instances (e.g., m5.xlarge).
- **The sizing recommendations provided by this tool focus exclusively on the **worker nodes**, as these are the nodes you provision and which run your application workloads.** The cost and capacity of managed control plane/infra nodes are not included in the recommendations.

### Minimum Requirements

- A **minimum of 3 worker nodes** distributed across Availability Zones is strongly recommended for production ROSA clusters to ensure high availability, resilience against node failures, and sufficient capacity for core OpenShift components that may run on workers. This tool enforces this minimum.
- Using multiple Availability Zones (AZs) is critical for a production-ready ROSA cluster architecture.

### Instance Types

- Recommendations are generated by evaluating a built-in list of current and recent generation x86_64 AWS instance types supported by ROSA.
- The list includes:
    - **General Purpose (m-series, m7i, m6i, m7g):** Provide a balance of compute, memory, and networking resources. Good for a wide range of applications. Includes Intel and Graviton (ARM) variants. Bare metal options (.metal) are included.
    - **Compute Optimized (c-series, c7i, c7g):** Offer a higher ratio of vCPUs to memory, suitable for compute-intensive applications like batch processing, video encoding, or high-performance web servers. Includes Intel and Graviton (ARM) variants. Bare metal options (.metal) are included.
    - **Memory Optimized (r-series, r7i, r7g):** Offer a higher ratio of memory to vCPUs, suitable for memory-intensive applications like databases, real-time analytics, or caches. Includes Intel and Graviton (ARM) variants. Bare metal options (.metal) are included.
    - *(Note: Specialized instance types like Storage Optimized (i-series) or Accelerated Computing (p/g-series) are typically used for specific workloads and may require separate consideration or node pools not directly covered by the primary recommendation).*

### Sizing Algorithm & Utilization

- The algorithm determines the minimum node count of a given instance type needed to satisfy the calculated resource requirements (CPU, Memory, Pods, minimum HA).
- Utilization percentages are calculated based on the *required resources* (demand with redundancy) divided by the *total capacity* of the proposed number of nodes (supply).
- For very small workloads where the 3-node minimum is the primary driver for the node count, utilization will naturally appear low. This is expected and does not necessarily indicate an inefficient recommendation for minimal workloads requiring HA.
- For larger deployments, the efficiency scoring penalizes configurations with utilization far above or below the target (default 65%), aiming for a balance between cost efficiency and headroom. Configurations with extremely high utilization (>95%) are filtered out as they lack sufficient buffer.

## Cost Estimation

While the tool provides detailed instance type and node count recommendations, it does not include real-time AWS pricing. **It is essential to use the recommended configurations with the [AWS Pricing Calculator](https://aws.amazon.com/pricing/calculator/) to estimate the actual cost** based on your specific AWS region, desired purchasing options (On-Demand, Savings Plans, Reserved Instances), and any applicable ROSA fees.

## Contributing

Contributions welcome! Please submit Pull Requests with:

- Clear description of changes.
- Any relevant test cases.
- Documentation updates (like this README!).

## License

MIT License - see LICENSE file for details.