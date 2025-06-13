from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import os
from pathlib import Path
from importlib import resources
from importlib.resources import as_file


def get_packaged_executable_path(program_name: str) -> str:
    """Get the path to a packaged executable binary.
    
    Args:
        program_name: Name of the program ('euler', 'peaksearch', or 'pix2qs')
        
    Returns:
        Path to the executable
        
    Raises:
        FileNotFoundError: If the executable is not found or not executable
    """
    try:
        # Use importlib.resources to access the bundled executable
        bin_files = resources.files('laueanalysis.indexing.bin')
        exe_file = bin_files / program_name
        
        if not exe_file.is_file():
            raise FileNotFoundError(f"Executable '{program_name}' not found in package")
        
        # Return the path as a string
        return str(exe_file)
        
    except (ModuleNotFoundError, FileNotFoundError):
        raise FileNotFoundError(f"Could not locate executable '{program_name}' in laueanalysis.indexing.bin package")


@dataclass
class LaueConfig:
    """Configuration class for PyLaueGo."""
    # Output related parameters
    outputFolder: str = None
    filenamePrefix: str = None

    # Input related parameters
    filefolder: str = None
    configFile: str = None
    saveTxt: bool = False

    # Scan and depth range
    scanPointStart: Optional[int] = None
    scanPointEnd: Optional[int] = None
    depthRangeStart: Optional[int] = None
    depthRangeEnd: Optional[int] = None

    # Peak search parameters
    peaksearchPath: str = None
    boxsize: int = None
    maxRfactor: float = None
    min_size: int = None
    min_separation: int = None
    threshold: float = None
    thresholdRatio: float = -1
    peakShape: str = None
    max_peaks: int = None
    maskFile: str = None
    smooth: bool = False
    cosmicFilter: bool = False

    # P2Q parameters
    p2qPath: str = None
    geoFile: str = None
    crystFile: str = None

    # Indexing parameters
    indexingPath: str = None
    indexKeVmaxCalc: float = None
    indexKeVmaxTest: float = None
    indexAngleTolerance: float = None
    indexCone: float = None
    indexH: int = None
    indexK: int = None
    indexL: int = None

    # Additional parameters
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set default executable paths after initialization."""
        self._set_default_executable_paths()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LaueConfig':
        """Create a LaueConfig instance from a dictionary."""
        # Extract known fields
        known_fields = {k: config_dict.get(k) for k in cls.__annotations__ if k != 'extra' and k in config_dict}

        # Convert string values to appropriate types
        for field_name, field_value in known_fields.items():
            if field_value is None:
                continue

            field_type = cls.__annotations__.get(field_name)
            if field_type in (int, Optional[int]) and isinstance(field_value, str):
                try:
                    known_fields[field_name] = int(field_value)
                except (ValueError, TypeError):
                    pass
            elif field_type in (float, Optional[float]) and isinstance(field_value, str):
                try:
                    known_fields[field_name] = float(field_value)
                except (ValueError, TypeError):
                    pass
            elif field_type is bool and isinstance(field_value, str):
                known_fields[field_name] = field_value.lower() in ('true', 't', 'yes', 'y', '1')

        # Store unknown fields in extra
        extra_fields = {k: v for k, v in config_dict.items() if k not in cls.__annotations__}

        instance = cls(**known_fields, extra=extra_fields)
        return instance

    def get(self, key, default=None):
        """Get a configuration value by key."""
        if hasattr(self, key):
            return getattr(self, key)
        return self.extra.get(key, default)

    def __getattr__(self, name):
        """Allow falling back to the extra dict for unknown attributes."""
        if name in self.extra:
            return self.extra[name]
        raise AttributeError(f"'LaueConfig' object has no attribute '{name}'")
    
    def _set_default_executable_paths(self):
        """Set default paths to packaged executables if not already set."""
        if not self.peaksearchPath:
            self.peaksearchPath = get_packaged_executable_path('peaksearch')
        if not self.p2qPath:
            self.p2qPath = get_packaged_executable_path('pix2qs')  # Note: executable is 'pix2qs'
        if not self.indexingPath:
            self.indexingPath = get_packaged_executable_path('euler')
