"""Configuration for the public MSF-Net implementation."""
from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Config:
    # Set MSFNET_DATA_ROOT to the directory containing ehcc/ and hgdn/.
    # Example (PowerShell): $env:MSFNET_DATA_ROOT = "D:\HSI_Reflectance"
    data_root: Path = Path(os.environ.get("MSFNET_DATA_ROOT", "./data")).expanduser()
    ehcc_folder: str = "ehcc"
    hgdn_folder: str = "hgdn"
    project_root: Path = Path(__file__).resolve().parent
    output_folder: str = "MSF-Net-Preprocessed"
    patch_size: int = 100
    grid_size: int = 5
    selected_ehcc: int = 30
    selected_hgdn: int = 30
    # The submitted manuscript (Model Training and Experimental Settings) specifies 2026.
    seed: int = 2026
    outer_folds: int = 5
    inner_folds: int = 4
    rgb_wavelengths_nm = (650.0, 550.0, 450.0)
    max_rgb_band_distance_nm: float = 4.0
    use_region_snv: bool = True
    use_mean_spectrum_snv: bool = False
    eps: float = 1e-6
    rgb_clip_min: float = 0.0
    rgb_clip_max: float = 1.0
    cache_folder: str = "cache_region25_seed2026"
    batch_size: int = 4
    num_workers: int = 0
    use_amp: bool = True

    @property
    def output_dir(self) -> Path:
        return self.data_root / self.output_folder

    @property
    def cache_dir(self) -> Path:
        return self.output_dir / self.cache_folder


CFG = Config()
