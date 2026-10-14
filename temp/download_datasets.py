#!/usr/bin/env python3
"""Fetch publisher-hosted datasets, audit received bytes, and create a ZIP64 archive.

Does not generate fake data, infer missing labels, or alter publisher splits.
Requires network access; Roboflow exports usually require an API key.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED

ROOT = Path(__file__).resolve().parent
IMAGES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
ANNOTATIONS = {'.txt', '.json', '.xml', '.yaml', '.yml', '.csv', '.png'}
SKIP_DIRS = {'.git', '__pycache__', '.ipynb_checkpoints', '.venv'}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def all_files(directory):
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob('*') if p.is_file() and not any(part in SKIP_DIRS for part in p.relative_to(directory).parts))


def inspect_dataset(directory):
    files = all_files(directory)
    is_mask = lambda p: p.stem.lower().endswith('_mask')
    images = [p for p in files if p.suffix.lower() in IMAGES and not is_mask(p)]
    ann = [p for p in files if (p.suffix.lower() in ANNOTATIONS and p.suffix.lower() not in IMAGES) or is_mask(p)]
    archives = [p for p in files if p.suffix.lower() == '.zip']
    inner_imgs = 0
    bad_archives = []
    for p in archives:
        try:
            with ZipFile(p) as z:
                if z.testzip() is not None:
                    bad_archives.append(str(p.relative_to(directory)))
                else:
                    inner_imgs += sum(Path(n).suffix.lower() in IMAGES for n in z.namelist())
        except Exception:
            bad_archives.append(str(p.relative_to(directory)))
    return {
        'files': len(files), 'images_on_disk': len(images),
        'images_inside_archives': inner_imgs, 'nonimage_annotation_candidates': len(ann),
        'archive_files': len(archives), 'corrupt_archives': bad_archives,
        'bytes': sum(p.stat().st_size for p in files)
    }


def download_rf(spec, dest, api_key):
    if not api_key:
        raise RuntimeError('Roboflow API key missing. Set ROBOFLOW_API_KEY or use --roboflow-api-key.')
    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise RuntimeError('Install dependencies: python -m pip install -r requirements.txt') from exc
    dest.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=api_key)
    version = rf.workspace(spec['workspace']).project(spec['project']).version(int(spec['version']))
    version.download(spec['format'], location=str(dest), overwrite=True)


def download_git(spec, dest):
    git = shutil.which('git')
    if not git:
        raise RuntimeError('git CLI required for AGDD. Install Git from https://git-scm.com/downloads')
    if dest.exists() and (dest/'.git').exists():
        cmd = [git, '-C', str(dest), 'pull', '--ff-only']
    elif dest.exists() and any(dest.iterdir()):
        raise RuntimeError('Destination exists and is not a Git checkout: ' + str(dest))
    else:
        cmd = [git, 'clone', '--depth', '1', '--branch', spec.get('branch', 'main'), spec['repo'], str(dest)]
    subprocess.run(cmd, check=True)
    lf = subprocess.run([git, '-C', str(dest), 'lfs', 'version'], capture_output=True, text=True)
    if lf.returncode == 0:
        subprocess.run([git, '-C', str(dest), 'lfs', 'pull'], check=True)
    else:
        print('Warning: git-lfs not installed; check for LFS pointers, if used.')


def download_gdrive(spec, dest):
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-m', 'gdown', '--continue', '--retries', '3', spec['url'], '-O', str(dest)]
    subprocess.run(cmd, check=True)


def package_zip(root, outfile, included_ids):
    files = all_files(root)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(outfile, 'w', allowZip64=True) as z:
        for p in files:
            if p.resolve() == outfile.resolve() or p.name.endswith('.part'):
                continue
            relative = p.relative_to(root)
            if relative.parts and relative.parts[0] == 'datasets' and len(relative.parts) > 1 and relative.parts[1] not in included_ids:
                continue
            compress = ZIP_STORED if p.suffix.lower() in IMAGES | {'.zip', '.7z', '.mp4'} else ZIP_DEFLATED
            z.write(p, arcname=str(relative).replace(os.sep, '/'), compress_type=compress)
    with ZipFile(outfile) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError('Package verification failed for member: ' + bad)
    return {'zip': str(outfile.resolve()), 'sha256': sha256(outfile), 'bytes': outfile.stat().st_size}


def load_previous_records(status_path, replacing_ids):
    """Records from an earlier run for datasets not being re-run, so --only does not erase them."""
    try:
        old = json.loads(Path(status_path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return []
    return [r for r in old.get('records', []) if r.get('id') not in replacing_ids]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', default='glass_crack_dataset_bundle', help='Destination directory')
    ap.add_argument('--only', default='all', help='Comma-separated dataset IDs, or all')
    ap.add_argument('--roboflow-api-key', default=os.getenv('ROBOFLOW_API_KEY'), help='DO NOT commit or share your key')
    ap.add_argument('--audit-only', action='store_true', help='Do not download; audit existing dataset folder and package it')
    ap.add_argument('--no-zip', action='store_true', help='Skip final ZIP file')
    args = ap.parse_args(argv)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output/'datasets').mkdir(exist_ok=True)
    catalog = json.loads((ROOT/'datasets.json').read_text(encoding='utf-8'))
    specs = catalog['datasets']
    wanted = {s['id'] for s in specs} if args.only == 'all' else set(args.only.split(','))
    unknown = wanted - {s['id'] for s in specs}
    if unknown:
        ap.error('Unknown dataset IDs: '+', '.join(sorted(unknown)))
    shutil.copy2(ROOT/'datasets.json', output/'datasets.json')
    previous = load_previous_records(output/'download_status.json', wanted)
    status = {'generated_utc': utc_now(), 'network_download_attempted': not args.audit_only,
              'records': list(previous), 'notes': 'Presence of images is not independent defect-label validation.'}
    print('Output:', output)
    for spec in specs:
        if spec['id'] not in wanted:
            continue
        dest = output/'datasets'/spec['id']
        rec = {'id': spec['id'], 'source': spec['source'], 'license_reported': spec['license_reported'],
               'reported_image_count': spec['publisher_reported_images'],
               'run_started_utc': utc_now()}
        err = None
        try:
            if not args.audit_only:
                print('\nDownloading', spec['name'], '...', flush=True)
                if spec['method'] == 'roboflow':
                    download_rf(spec, dest, args.roboflow_api_key)
                elif spec['method'] == 'github':
                    download_git(spec, dest)
                elif spec['method'] == 'gdrive':
                    download_gdrive(spec, dest)
            stats = inspect_dataset(dest)
            rec['inspection'] = stats
            if stats['corrupt_archives']:
                rec['status'] = 'INVALID_ARCHIVE'
            elif stats['images_on_disk'] + stats['images_inside_archives'] == 0:
                rec['status'] = 'NO_IMAGE_BYTES'
                if not args.audit_only:
                    err = 'No images found after purported download'
            else:
                rec['status'] = 'FILES_PRESENT_NOT_LABEL_VERIFIED'
        except Exception as ex:
            err = f'{type(ex).__name__}: {ex}'
            rec['status'] = 'DOWNLOAD_FAILED'
            rec['inspection'] = inspect_dataset(dest)
        if err:
            rec['error'] = err
            print('  SKIPPED/FAILED:', err, file=sys.stderr)
        else:
            print('  status:', rec['status'], '| images:', rec['inspection']['images_on_disk'],
                  '| zipped images:', rec['inspection']['images_inside_archives'])
        rec['run_finished_utc'] = utc_now()
        status['records'].append(rec)
        # Incremental crash-safe manifest; keys never written.
        (output/'download_status.json').write_text(json.dumps(status, indent=2)+'\n',encoding='utf-8')
    hash_csv = output/'files_sha256.csv'
    kept_rows = []
    if hash_csv.exists():
        with hash_csv.open(newline='',encoding='utf-8') as f:
            kept_rows = [r for r in csv.reader(f) if r and r[0] != 'dataset_id' and r[0] not in wanted]
    with hash_csv.open('w',newline='',encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['dataset_id','relative_path','bytes','sha256'])
        wr.writerows(kept_rows)
        for spec in specs:
            if spec['id'] not in wanted: continue
            for p in all_files(output/'datasets'/spec['id']):
                wr.writerow([spec['id'],p.relative_to(output).as_posix(),p.stat().st_size,sha256(p)])
    order = [s['id'] for s in specs]
    status['records'].sort(key=lambda r: order.index(r['id']) if r['id'] in order else len(order))
    successful = [r['id'] for r in status['records'] if r['status'] == 'FILES_PRESENT_NOT_LABEL_VERIFIED']
    status['successful_datasets'] = successful
    status['dataset_count'] = len(successful)
    (output/'download_status.json').write_text(json.dumps(status, indent=2)+'\n',encoding='utf-8')
    run_ok = [i for i in successful if i in wanted]
    if not args.no_zip:
        if run_ok:
            result = package_zip(output, output.parent/'glass_crack_datasets.zip',set(wanted))
            (output.parent/'glass_crack_datasets.zip.sha256').write_text(result['sha256']+'  '+Path(result['zip']).name+'\n',encoding='utf-8')
            print('\nCreated:', result['zip'], f"({result['bytes']:,} bytes)")
        else:
            print('\nNo downloaded dataset images found; refusing to create an EMPTY dataset ZIP.',file=sys.stderr)
    print('Manifest:', output/'download_status.json')
    return 0 if run_ok and len(run_ok) == len(wanted) else 2


if __name__ == '__main__':
    raise SystemExit(main())
