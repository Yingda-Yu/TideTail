"""Global coastal data source registry and discovery.

Enumerates publicly-available datasets relevant to TideTail's extreme
coastal water-level forecasting goal. Each source is tracked through
a state machine:
    DISCOVERED → VERIFIED → LICENSE_CHECKED → DOWNLOADED → VALIDATED → ADMITTED

This module does NOT download data — it is the registry only. Download
logic lives in src/tidetail/data_sources/*.py.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class SourceStatus(str, Enum):
    DISCOVERED = "DISCOVERED"
    VERIFIED = "VERIFIED"
    LICENSE_CHECKED = "LICENSE_CHECKED"
    DOWNLOADED = "DOWNLOADED"
    VALIDATED = "VALIDATED"
    ADMITTED = "ADMITTED"
    EXCLUDED = "EXCLUDED"
    AUTH_REQUIRED = "AUTH_REQUIRED"


@dataclass
class DataSource:
    """One registered data source."""

    source_id: str
    provider: str
    dataset_name: str
    url: str
    variables: list[str] = field(default_factory=list)
    coverage_region: str = ""
    coverage_time_start: str = ""
    coverage_time_end: str = ""
    resolution_time: str = ""
    resolution_space: str = ""
    file_format: str = ""
    license: str = ""
    redistribution_restrictions: str = ""
    auth_required: bool = False
    auth_hint: str = ""
    download_method: str = ""
    retrieval_status: str = SourceStatus.DISCOVERED.value
    download_size_estimate_gb: float = 0.0
    local_path: str = ""
    sha256: str = ""
    priority: int = 3  # 1=highest, 3=normal, 5=low
    citation: str = ""
    notes: str = ""


# ==========================================================================
# Registry — compiled from public knowledge, subject to verification
# ==========================================================================

REGISTRY: list[DataSource] = [
    # --- Water level observations ---
    DataSource(
        source_id="noaa_coops",
        provider="NOAA CO-OPS",
        dataset_name="Center for Operational Oceanographic Products and Services",
        url="https://api.tidesandcurrents.noaa.gov/api/prod/",
        variables=["observed_total_water_level", "astronomical_tide", "residual", "wind", "water_temperature"],
        coverage_region="North America (US coasts), some Caribbean, Atlantic islands",
        coverage_time_start="1990-01-01 (older records available)",
        coverage_time_end="present",
        resolution_time="6-min / hourly",
        file_format="CSV, JSON via API",
        license="Public domain (US government)",
        redistribution_restrictions="Cite NOAA CO-OPS, data may not be used for navigational safety",
        auth_required=False,
        download_method="REST API (https://api.tidesandcurrents.noaa.gov/api/prod/)",
        retrieval_status=SourceStatus.VERIFIED.value,
        download_size_estimate_gb=0.001,
        priority=1,
        citation="NOAA CO-OPS API Documentation. https://api.tidesandcurrents.noaa.gov/api/prod/",
        notes="Primary NOAA US water-level + tide API. No auth needed. Hourly extremes available. Verified 2026-10-31: station 8518750 (The Battery, NYC), 1680 rows/7 days.",
    ),
    DataSource(
        source_id="gesla3",
        provider="GESLA",
        dataset_name="Global Extreme Sea Level Analysis v3",
        url="https://www.gesla.org/",
        variables=["observed_total_water_level", "tide", "surge_residual"],
        coverage_region="Global (2000+ gauges worldwide)",
        coverage_time_start="1880-01-01 (varies by gauge)",
        coverage_time_end="2023",
        resolution_time="Hourly",
        file_format="ASCII/GEO (NetCDF supplement)",
        license="CC BY 4.0",
        redistribution_restrictions="Cite GESLA, co-authorship if used substantially",
        auth_required=False,
        download_method="Direct download from gesla.org (requires registration form, no API key)",
        priority=1,
        citation="Haigh, I.D., et al. (2023). GESLA-3: Global Extreme Sea Level Analysis.",
    ),
    DataSource(
        source_id="psmsl",
        provider="PSMSL",
        dataset_name="Permanent Service for Mean Sea Level",
        url="https://www.psmsl.org/",
        variables=["sea_level_monthly_mean", "tide_gauges"],
        coverage_region="Global (1400+ stations)",
        coverage_time_start="1880 (older)",
        coverage_time_end="present",
        resolution_time="Monthly means (primary), hourly via tide gauge links",
        license="CC BY 4.0",
        redistribution_restrictions="Cite PSMSL",
        auth_required=False,
        download_method="Direct download after registration",
        priority=3,
        notes="Monthly means useful for long-term background but not directly for storm peaks. Hourly tide gauge links exist.",
    ),
    DataSource(
        source_id="u_hs_lc",
        provider="UHSLC",
        dataset_name="University of Hawaii Sea Level Center",
        url="https://uhslc.soest.hawaii.edu/",
        variables=["hourly_sea_level", "meteorological_conditions"],
        coverage_region="Pacific, Indian Ocean, global",
        coverage_time_start="1990 (hourly)",
        coverage_time_end="present",
        resolution_time="Hourly",
        license="Free for research, cite UHSLC",
        auth_required=False,
        download_method="FTP download after registration",
        priority=2,
    ),
    DataSource(
        source_id="emodnet_physics",
        provider="EMODnet",
        dataset_name="EMODnet Physics In-situ Observations",
        url="https://physics.emodnet.eu/",
        variables=["water_level", "temperature", "salinity", "current"],
        coverage_region="Europe (North Sea, Baltic, Mediterranean, Black Sea)",
        coverage_time_start="2000 (varies)",
        resolution_time="Variable (hourly typical)",
        license="CC BY 4.0",
        auth_required=False,
        download_method="Download via portal, API available",
        priority=1,
    ),
    DataSource(
        source_id="bom_australia",
        provider="Bureau of Meteorology (Australia)",
        dataset_name="Australian Tide Gauge Network",
        url="https://www.bom.gov.au/oceanography/projects/ntc/databrowser.shtml",
        variables=["observed_total_water_level", "astronomical_tide", "storm_surge"],
        coverage_region="Australia (35+ gauges)",
        coverage_time_start="1994 (continuous)",
        coverage_time_end="present",
        resolution_time="Hourly (some higher)",
        license="Public, cite BOM",
        auth_required=False,
        priority=2,
    ),
    DataSource(
        source_id="jma_japan",
        provider="Japan Meteorological Agency",
        dataset_name="Japan Storm Surge and Tide Observations",
        url="https://www.data.jma.go.jp/kaiyou/data/db/storm_surge/index.html",
        coverage_region="Japan coast",
        license="Public access (some Japanese-only portals)",
        priority=2,
        auth_required=False,
    ),

    # --- Meteorological forcing ---
    DataSource(
        source_id="era5_single_levels",
        provider="ECMWF / Copernicus Climate Change Service",
        dataset_name="ERA5 Hourly Single Levels Reanalysis",
        url="https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels",
        variables=["10m_u_wind", "10m_v_wind", "mean_sea_level_pressure", "surface_pressure",
                   "precipitation", "wind_gust"],
        coverage_region="Global (0.25° grid)",
        coverage_time_start="1950-01-01",
        coverage_time_end="near-real-time (with latency)",
        resolution_time="Hourly",
        resolution_space="0.25° (~25 km at equator)",
        file_format="NetCDF, GRIB",
        license="Copernicus license (free for research, attribution required)",
        auth_required=True,
        auth_hint="CDS API key: register at cds.climate.copernicus.eu",
        download_method="CDS API (cdsapi Python client)",
        download_size_estimate_gb=0.5,
        priority=1,
        notes="CRITICAL: ERA5 is a REANALYSIS (hindcast), not a forecast archive. For operational forecast experiments, use NOAA GFS operational archives. Using ERA5 as if it were forecast = leakage.",
    ),
    DataSource(
        source_id="noaa_hurdat2",
        provider="NOAA National Hurricane Center",
        dataset_name="Atlantic Hurricane Database HURDAT2",
        url="https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2025-02272026.txt",
        variables=["storm_position", "storm_intensity_pressure", "storm_intensity_windspeed", "trajectory"],
        coverage_region="Atlantic Ocean (including Caribbean, Gulf of Mexico)",
        coverage_time_start="1851",
        coverage_time_end="2025",
        resolution_time="6-hourly",
        file_format="TXT (parsable)",
        license="Public domain (US government)",
        auth_required=False,
        download_method="Direct file download",
        download_size_estimate_gb=0.007,
        retrieval_status=SourceStatus.VERIFIED.value,
        priority=1,
        citation="Landsea, C.W. et al. (2014). The Revised Atlantic Hurricane Database.",
        notes="Atlantic tropical cyclone tracks and intensities. 2004 storms, 55605 data points (verified 2026-10-31).",
    ),
    DataSource(
        source_id="ibtracs",
        provider="NOAA / IBTrACS Project",
        dataset_name="International Best Track Archive for Climate Stewardship",
        url="https://www.ncei.noaa.gov/products/international-best-track-archive-for-climate-stewardship",
        variables=["storm_position", "intensity_windspeed", "pressure", "trajectory"],
        coverage_region="Global tropical basins",
        coverage_time_start="1842",
        coverage_time_end="present",
        resolution_time="3/6/12-hourly",
        file_format="NetCDF, CSV",
        license="Public domain (NOAA)",
        auth_required=False,
        priority=1,
        notes="GLOBAL tropical cyclone tracks. Primary source for storm event labeling in all regions.",
    ),

    # --- Ocean physical fields ---
    DataSource(
        source_id="copernicus_ocean_reanalysis",
        provider="Copernicus Marine Service",
        dataset_name="Global Ocean Physics Reanalysis (GLORYS)",
        url="https://marine.copernicus.eu/product-detail/global-reanalysis-phy-001-030-daily",
        variables=["sea_surface_height", "sea_surface_height_anomaly", "ocean_current",
                   "temperature", "salinity"],
        coverage_region="Global (1/12° grid)",
        coverage_time_start="1993-01-01",
        coverage_time_end="present",
        resolution_time="Daily",
        resolution_space="1/12° (~8 km)",
        file_format="NetCDF",
        license="Copernicus license",
        auth_required=True,
        auth_hint="CMEMS/Motu API key",
        priority=2,
    ),
    DataSource(
        source_id="gtsm_era5",
        provider="TU Delft / Deltares",
        dataset_name="Global Tide and Surge Model (GTSM) forced by ERA5",
        url="https://github.com/Deltares-research/GTSM",
        variables=["water_level", "storm_surge", "tide"],
        coverage_region="Global (1/4° grid)",
        resolution_time="Hourly",
        resolution_space="0.25°",
        file_format="NetCDF",
        license="Apache 2.0 (code), data may be restricted",
        auth_required=True,
        auth_hint="Requires CMEMS/ERA5 access for forcing; some GTSM runs available via open repositories",
        priority=2,
        notes="Global tide + surge model. Provides simulated total water level at coastal points.",
    ),

    # --- Bathymetry / static ---
    DataSource(
        source_id="gebco",
        provider="GEBCO (General Bathymetric Chart of the Oceans)",
        dataset_name="GEBCO Gridded Bathymetry",
        url="https://www.gebco.net/data_and_products/gridded_bathymetry_data/",
        variables=["bathymetry", "elevation"],
        coverage_region="Global ocean + coastal",
        resolution_space="15 arc-second (~400m near equator)",
        file_format="NetCDF, GeoTIFF",
        license="CC BY 4.0",
        auth_required=True,
        auth_hint="Registration at gebco.net",
        priority=2,
        notes="Coastal bathymetry for physical constraints and storm surge modeling.",
    ),
]


def registry_to_dicts() -> list[dict[str, Any]]:
    return [asdict(s) for s in REGISTRY]


def save_registry(path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(registry_to_dicts(), fh, indent=2, default=str)


def priority_sources() -> list[DataSource]:
    """Return priority-1 sources, sorted by current status."""
    return sorted(
        [s for s in REGISTRY if s.priority == 1],
        key=lambda s: (
            s.retrieval_status.value if isinstance(s.retrieval_status, SourceStatus)
            else str(s.retrieval_status),
            s.source_id,
        ),
    )


if __name__ == "__main__":
    save_registry("data/catalog/global_sources.json")
    print(f"Registry: {len(REGISTRY)} sources")
    print(f"Priority-1: {len(priority_sources())}")
    for s in priority_sources():
        print(f"  [{s.retrieval_status}] {s.source_id} — {s.dataset_name[:50]}")
