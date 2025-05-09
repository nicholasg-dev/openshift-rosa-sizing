# Changelog

All notable changes to the OpenShift ROSA Sizing Tool will be documented in this file.

## [2.4.0] - 2025-04-27

### Changed

- Improved efficiency scoring for small workloads where the 3-node minimum HA requirement is enforced
- Reduced the utilization deviation penalty for small workloads to better reflect expected low utilization
- Added more detailed explanation in the general notes about the adjusted scoring for small workloads
- Updated internal code to pass the theoretical node count (before HA minimum) to the scoring function

## [2.1.0] - 2025-04-26

### Added

- Improved handling of small workloads with 3-node minimum HA requirement
- Added "HA Minimum" as a new limiting factor to clearly indicate when the 3-node minimum is the limiting factor
- Enhanced efficiency score calculation for small workloads to be more lenient with low utilization
- Updated text report with additional notes about utilization expectations for small workloads

### Changed

- Modified utilization filtering logic to remove the lower bound (previously 20%) for minimum HA deployments
- Maintained the upper bound of 80% utilization for all deployments to prevent overloading
- For larger deployments (more than 3 nodes), maintained the 30-80% utilization target range
- Updated README.md with information about the utilization filtering changes

### Fixed

- Fixed issue where small workloads could be filtered out due to low utilization when the 3-node minimum is enforced

## [2.0.0] - 2025-04-01

### Added

- Real data collection from OpenShift clusters
- Enhanced documentation with complete code comments and usage examples
- ROSA-specific logic including managed control plane considerations
- Cost estimation with integrated rough cost projections
- Workload profiling with automatic CPU/Memory-bound detection
- Comprehensive error handling and validation

### Changed (1)

- Removed all sample data dependencies
- Improved algorithm for instance type selection
- Enhanced reporting format with more detailed explanations

### Fixed (1)

- Various bug fixes and performance improvements
