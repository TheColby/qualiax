"""Model assets: pinned download URLs, SHA-256 checksums, and the local model directory.

Model files are never bundled with qualiax. ``qualiax models download`` (or
:func:`download_models`) fetches them into :func:`model_dir` and refuses any file
whose size or SHA-256 does not match the pinned value, so every run with a model
uses exactly the weights the checksums describe.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

_DNS_CHALLENGE_REV = "591184a9fcb2cbdec02520fed81a32bbbf9d73ff"
_DNS_CHALLENGE_RAW = f"https://raw.githubusercontent.com/microsoft/DNS-Challenge/{_DNS_CHALLENGE_REV}"


@dataclass(frozen=True)
class ModelAsset:
    name: str
    relative_path: str
    url: str
    sha256: str
    size: int
    license: str
    source: str
    description: str


MODEL_ASSETS: dict[str, ModelAsset] = {
    asset.name: asset
    for asset in (
        ModelAsset(
            name="dnsmos-p835",
            relative_path="dnsmos/sig_bak_ovr.onnx",
            url=f"{_DNS_CHALLENGE_RAW}/DNSMOS/DNSMOS/sig_bak_ovr.onnx",
            sha256="269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd",
            size=1_157_965,
            license="CC-BY-4.0",
            source=f"microsoft/DNS-Challenge@{_DNS_CHALLENGE_REV[:12]}",
            description="Microsoft DNSMOS P.835 (SIG, BAK, OVRL)",
        ),
        ModelAsset(
            name="dnsmos-p808",
            relative_path="dnsmos/model_v8.onnx",
            url=f"{_DNS_CHALLENGE_RAW}/DNSMOS/DNSMOS/model_v8.onnx",
            sha256="9246480c58567bc6affd4200938e77eef49468c8bc7ed3776d109c07456f6e91",
            size=224_860,
            license="CC-BY-4.0",
            source=f"microsoft/DNS-Challenge@{_DNS_CHALLENGE_REV[:12]}",
            description="Microsoft DNSMOS P.808 overall MOS",
        ),
    )
}


class ModelAssetError(RuntimeError):
    """A model file could not be downloaded or failed verification."""


def model_dir() -> Path:
    """``$QUALIAX_MODEL_DIR``, else ``$XDG_CACHE_HOME/qualiax/models``, else ``~/.cache/qualiax/models``."""
    override = os.environ.get("QUALIAX_MODEL_DIR")
    if override:
        return Path(override).expanduser()
    cache_home = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
    return base / "qualiax" / "models"


def asset_path(name: str, directory: str | Path | None = None) -> Path:
    return Path(directory or model_dir()) / MODEL_ASSETS[name].relative_path


_VERIFIED: dict[tuple[str, int, int], bool] = {}


def asset_status(name: str, directory: str | Path | None = None) -> str:
    """``"ok"``, ``"missing"``, or ``"checksum_mismatch"`` for a registered asset."""
    path = asset_path(name, directory)
    if not path.is_file():
        return "missing"
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _VERIFIED:
        _VERIFIED[key] = stat.st_size == MODEL_ASSETS[name].size and _sha256(path) == MODEL_ASSETS[name].sha256
    return "ok" if _VERIFIED[key] else "checksum_mismatch"


def verified_asset_path(name: str, directory: str | Path | None = None) -> Path | None:
    """The asset's path if it is present and matches its pinned checksum, else ``None``."""
    return asset_path(name, directory) if asset_status(name, directory) == "ok" else None


def verify_models(directory: str | Path | None = None) -> list[dict]:
    return [
        {**_describe(asset), "path": str(asset_path(asset.name, directory)), "status": asset_status(asset.name, directory)}
        for asset in MODEL_ASSETS.values()
    ]


def download_models(
    names: Iterable[str] | None = None,
    directory: str | Path | None = None,
    *,
    force: bool = False,
    opener: Callable = urllib.request.urlopen,
) -> list[dict]:
    """Fetch registered assets, verifying size and SHA-256 before anything is kept.

    Files that already verify are left alone unless ``force`` is set. A download
    that doesn't match its pinned checksum raises :class:`ModelAssetError` and
    leaves no file behind.
    """
    selected = list(names) if names is not None else list(MODEL_ASSETS)
    unknown = sorted(set(selected) - set(MODEL_ASSETS))
    if unknown:
        raise ModelAssetError(f"Unknown model asset(s): {', '.join(unknown)}. Known: {', '.join(MODEL_ASSETS)}")
    report = []
    for name in selected:
        asset = MODEL_ASSETS[name]
        path = asset_path(name, directory)
        if not force and asset_status(name, directory) == "ok":
            report.append({**_describe(asset), "path": str(path), "status": "present"})
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        digest, size = hashlib.sha256(), 0
        try:
            with opener(asset.url, timeout=60) as response, partial.open("wb") as handle:
                for chunk in iter(lambda: response.read(1 << 16), b""):
                    digest.update(chunk)
                    size += len(chunk)
                    handle.write(chunk)
            if size != asset.size or digest.hexdigest() != asset.sha256:
                raise ModelAssetError(
                    f"{name}: downloaded {size} bytes with SHA-256 {digest.hexdigest()}, "
                    f"expected {asset.size} bytes with SHA-256 {asset.sha256}"
                )
            os.replace(partial, path)
        except ModelAssetError:
            partial.unlink(missing_ok=True)
            raise
        except Exception as exc:
            partial.unlink(missing_ok=True)
            raise ModelAssetError(f"{name}: download from {asset.url} failed: {exc}") from exc
        report.append({**_describe(asset), "path": str(path), "status": "downloaded"})
    return report


def _describe(asset: ModelAsset) -> dict:
    described = asdict(asset)
    described.pop("relative_path")
    return described


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
