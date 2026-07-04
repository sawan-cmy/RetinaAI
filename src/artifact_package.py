from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime
from pathlib import Path


def _add_matches(bundle: zipfile.ZipFile, pattern: str, root: Path) -> int:
    count = 0
    for path in root.glob(pattern):
        if path.is_file():
            bundle.write(path, path.as_posix())
            count += 1
    return count


def package_training_run(
    model_name: str,
    reports_dir: str | Path = "reports",
    screenshots_dir: str | Path = "docs/screenshots",
    output_dir: str | Path = "reports/packages",
    package_name: str | None = None,
) -> Path:
    reports_dir = Path(reports_dir)
    screenshots_dir = Path(screenshots_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_path = output_dir / (package_name or f"{model_name}_{timestamp}_metrics_package.zip")

    manifest: dict[str, int | str] = {"model_name": model_name, "created_at": timestamp}
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for pattern in [
            "*.json",
            "*.csv",
            "*.png",
            "figures/*",
            "calibration/*",
            "cards/*",
            "external_validation/*",
        ]:
            manifest[pattern] = _add_matches(bundle, pattern, reports_dir)
        if screenshots_dir.exists():
            manifest["screenshots"] = _add_matches(bundle, "*", screenshots_dir)
        bundle.writestr("manifest.json", json.dumps(manifest, indent=2, allow_nan=False))
    return package_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Package RetinaAI screenshots, cards, calibration, and final metrics.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--screenshots-dir", default="docs/screenshots")
    parser.add_argument("--out-dir", default="reports/packages")
    args = parser.parse_args()
    path = package_training_run(args.model, args.reports_dir, args.screenshots_dir, args.out_dir)
    print(json.dumps({"package_path": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())