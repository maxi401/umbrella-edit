#!/usr/bin/env python3
"""Make a private archive for one-time transfer to the hosted persistent disk."""
import argparse
import sqlite3
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('destination', type=Path, help='Private output .tar.gz file, outside the Git repository')
parser.add_argument('--source', type=Path, default=Path.home() / 'Desktop' / 'fashionArchive')
args = parser.parse_args()
destination = args.destination.expanduser().resolve()
if ROOT == destination or ROOT in destination.parents:
    parser.error('Choose an output path outside the project directory')
if destination.exists():
    parser.error('Destination already exists; choose a new file')
source = args.source.expanduser().resolve()
if not source.is_dir():
    parser.error('The original fashionArchive folder is unavailable')
with tempfile.TemporaryDirectory() as tmp:
    snapshot = Path(tmp) / 'fashion_archive.sqlite3'
    with sqlite3.connect(ROOT / 'fashion_archive.sqlite3') as old, sqlite3.connect(snapshot) as new:
        old.backup(new)
    with tarfile.open(destination, 'x:gz') as archive:
        archive.add(snapshot, arcname='fashion_archive.sqlite3')
        archive.add(source, arcname='source', recursive=True)
        for name in ('uploads', 'output'):
            path = ROOT / name
            if path.exists():
                archive.add(path, arcname=name, recursive=True)
print(f'Private archive created: {destination}')
