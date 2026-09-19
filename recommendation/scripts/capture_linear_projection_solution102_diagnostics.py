#!/usr/bin/env python3
"""Capture cap256/BN1/workspace1 training with one private DW catalog redirect.

Only the BF16 Ailk_Bjlk matching-table reference 103 is redirected to 102.
The existing capture, scalar queue and numerical implementations are reused.
Runtime kernel selection and training correctness still require observation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import capture_linear_projection_diagnostics as stages

base = stages.base
VARIANT = 'linear_projection_cap256_bn1_workspace1_dw_solution102_scalar_stages_v1'
DiagnosticScalarQueue = stages.DiagnosticScalarQueue
POLICY = stages.POLICY
PRIVATE_LIBRARY_PATH = '/data/mlperf_dlrm_v4/root_cause_20260919/dw_solution102_private_catalog_v1/gfx1250'
INSTALLED_LIBRARY_PATH = '/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250'
MANIFEST_PATH = str(Path(PRIVATE_LIBRARY_PATH).parent / 'manifest.json')
MANIFEST_SHA256 = '307ef5b7449920b06b2e549fb306d6f0a22bdb1bbdbfa7a19a825f79db5cf3cd'
TARGET_CATALOG_FILENAME = 'TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx1250.dat.zlib'
DERIVED_SHA256 = '3585e31939c0f2cf199d596b357c0f864aefcb29742ad13dd6ec6fb68432b962'
ENTRY_COUNT = 402
REQUIRED = dict(stages.mitigated.REQUIRED, HIPBLASLT_TENSILE_LIBPATH=PRIVATE_LIBRARY_PATH)
ABSENT = (*stages.mitigated.ABSENT, 'TENSILE_SOLUTION_INDEX', 'HIPBLASLT_TUNING_OVERRIDE_FILE')
FROZEN_SOURCES = {
    'capture_linear_projection_deferred.py': '2cf870a06afbe0ba04163583574c1bc48112cc9fbd752395bb76889163356ef5',
    'capture_linear_projection_mitigated.py': 'd3f1850a1f75a26645399e881f083b4cde067fb2325a0f69b798a36d295e25b6',
    'capture_linear_projection_diagnostics.py': '6a53a0c684144e60a16d716563ed5308304b35b9ee5113e5d95cb9a36fef6c57',
}
CONTROL_KEY = 'dw_solution102_private_catalog_control'
SCOPE = ('Only the selected BF16 Contraction_l_Ailk_Bjlk_Cijk_Dijk catalog matching table '
         'redirects 103 to 102 across varying sizes. Both solution definitions and all native '
         'objects remain unchanged. This is not a global VGPR policy or a validated training fix.')
LIMITS = [*stages.LIMITS, SCOPE,
          'The complete private catalog inventory is verified before and after bootstrap, '
          'at context archival and when a frame is saved. No extra numerical operation, '
          'queue flush or boundary readback is introduced.']


def file_pin(path):
    path = Path(path)
    before = path.stat()
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('Catalog file changed while hashing: ' + str(path))
    return {'path': str(path), 'bytes': after.st_size, 'sha256': sha}


def check_pin(spec):
    actual = file_pin(spec['path'])
    if actual != {name: spec[name] for name in ('path', 'bytes', 'sha256')}:
        raise ValueError('Catalog file identity mismatch: ' + spec['path'])
    return actual


def verify_private_catalog(environment):
    """Validate only files; all paths here are the actual worker's paths."""
    if environment.get('HIPBLASLT_TENSILE_LIBPATH') != PRIVATE_LIBRARY_PATH:
        raise ValueError('Worker lost the required private HIPBLASLT_TENSILE_LIBPATH')
    manifest_path = Path(MANIFEST_PATH)
    if manifest_path.is_symlink():
        raise ValueError('Private catalog manifest must be a regular file')
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256:
        raise ValueError('Private catalog manifest SHA256 mismatch')
    manifest = json.loads(manifest_bytes)
    private, installed = Path(PRIVATE_LIBRARY_PATH), Path(INSTALLED_LIBRARY_PATH)
    if (manifest.get('schema_version') != 1
            or manifest.get('private_library_path') != str(private)
            or manifest.get('installed_library_path') != str(installed)
            or manifest.get('target_catalog_filename') != TARGET_CATALOG_FILENAME
            or private == installed or private.is_symlink() or not private.is_dir()):
        raise ValueError('Private catalog manifest/path mismatch')
    rows = manifest['inventory']
    names = [row['name'] for row in rows]
    if (len(rows) != ENTRY_COUNT or len(set(names)) != ENTRY_COUNT
            or set(names) != {p.name for p in private.iterdir()}
            or set(names) != {p.name for p in installed.iterdir()}):
        raise ValueError('Private catalog inventory coverage mismatch')
    links, regular = 0, 0
    for row in rows:
        name = row['name']
        if Path(name).name != name:
            raise ValueError('Catalog inventory name is not a basename')
        original, chosen = installed / name, private / name
        if row['original']['path'] != str(original) or row['private']['path'] != str(chosen):
            raise ValueError('Catalog inventory path mismatch')
        if original.is_symlink() or not original.is_file():
            raise ValueError('Installed catalog entry is no longer a regular file')
        check_pin(row['original'])
        if name == TARGET_CATALOG_FILENAME:
            if (row['kind'] != 'derived_regular_file' or chosen.is_symlink()
                    or row['private']['sha256'] != DERIVED_SHA256):
                raise ValueError('Private DW catalog derivative identity mismatch')
            regular += 1
        else:
            if (row['kind'] != 'original_symlink' or not chosen.is_symlink()
                    or os.readlink(chosen) != str(original)
                    or row.get('symlink_target') != str(original)
                    or row['private']['sha256'] != row['original']['sha256']
                    or row['private']['bytes'] != row['original']['bytes']):
                raise ValueError('Private catalog symlink identity mismatch: ' + name)
            links += 1
        check_pin(row['private'])
    if (links, regular) != (ENTRY_COUNT - 1, 1):
        raise ValueError('Private catalog entry types changed')
    derived = check_pin(manifest['catalog']['derived'])
    if derived['path'] != str(private / TARGET_CATALOG_FILENAME) or derived['sha256'] != DERIVED_SHA256:
        raise ValueError('Private DW catalog record differs from inventory')
    master = check_pin(manifest['master']['private'])
    native = check_pin(manifest['native_dw_code_object']['private'])
    for key in ('master', 'native_dw_code_object'):
        left, right = manifest[key]['original'], manifest[key]['private']
        if (left['sha256'], left['bytes']) != (right['sha256'], right['bytes']):
            raise ValueError('Private master/native object differs from original')
    preparation = check_pin(manifest['preparation_source'])
    # Catch a file replaced during the complete inventory pass.
    if manifest_path.read_bytes() != manifest_bytes:
        raise ValueError('Private catalog manifest changed during verification')
    return {'scope': SCOPE, 'actual_private_library_path': environment['HIPBLASLT_TENSILE_LIBPATH'],
            'installed_library_path': str(installed),
            'manifest': {'path': str(manifest_path), 'bytes': len(manifest_bytes), 'sha256': MANIFEST_SHA256},
            'derived_catalog': derived, 'master': master, 'native_dw_code_object': native,
            'preparation_source': preparation, 'verified_inventory_files': ENTRY_COUNT,
            'original_symlinks': links, 'derived_regular_files': regular}


def verify_sources():
    inventory = stages.verify_sources()
    directory = Path(__file__).resolve().parent
    for name, expected in FROZEN_SOURCES.items():
        if file_pin(directory / name)['sha256'] != expected:
            raise base.BoundaryProbeError('Frozen solution102 diagnostic dependency changed: ' + name)
        inventory['scripts/' + name] = expected
    inventory['scripts/' + Path(__file__).name] = file_pin(__file__)['sha256']
    return inventory


def validate_environment(environment):
    for name, expected in REQUIRED.items():
        if environment.get(name) != expected:
            raise ValueError(f'Solution102 diagnostics requires {name}={expected}')
    for name in ABSENT:
        if name in environment:
            raise ValueError(f'Solution102 diagnostics requires {name} unset')
    return verify_private_catalog(environment)


def configure_environment(options, environment):
    control = validate_environment(environment)
    candidate = dict(environment)
    base._scoped(stages.configure_environment, verify_sources=verify_sources,
                 VARIANT=VARIANT)(options, candidate)
    recorded = json.loads(candidate['NAN_TRAINING_OPTIONS'])
    recorded[CONTROL_KEY] = control
    candidate['NAN_TRAINING_OPTIONS'] = json.dumps(recorded)
    if validate_environment(candidate) != control:
        raise ValueError('Private catalog changed during launch configuration')
    environment.update(candidate)


def validate_worker(options, environment):
    control = validate_environment(environment)
    base._scoped(stages.validate_worker, verify_sources=verify_sources,
                 VARIANT=VARIANT)(options, environment)
    if (json.loads(environment.get('NAN_TRAINING_OPTIONS', '{}')) != options
            or options.get(CONTROL_KEY) != control):
        raise ValueError('Solution102 worker lost private catalog provenance')
    return control


class Solution102DiagnosticProjectionProbe(stages.ProjectionProbe):
    def __init__(self, *args, catalog_bootstrap_verifications, **kwargs):
        self.catalog_bootstrap_verifications = copy.deepcopy(catalog_bootstrap_verifications)
        if [r['phase'] for r in self.catalog_bootstrap_verifications] != ['before_bootstrap', 'after_bootstrap']:
            raise ValueError('Private catalog requires both bootstrap verification records')
        base._scoped(stages.ProjectionProbe.__init__, verify_sources=verify_sources,
                     VARIANT=VARIANT, LIMITS=LIMITS)(self, *args, **kwargs)

    def save_frame(self, trigger):
        self.frame['metadata'][CONTROL_KEY] = verify_private_catalog(os.environ)
        self.frame['metadata']['private_catalog_bootstrap_verifications'] = copy.deepcopy(self.catalog_bootstrap_verifications)
        return base._scoped(stages.ProjectionProbe.save_frame,
                            VARIANT=VARIANT, LIMITS=LIMITS)(self, trigger)


ProjectionProbe = Solution102DiagnosticProjectionProbe


def split_capture(payload):
    metadata = payload.get('metadata', {})
    control = metadata.get(CONTROL_KEY, {})
    if (payload.get('capture_variant') != VARIANT
            or control.get('manifest', {}).get('sha256') != MANIFEST_SHA256
            or control.get('derived_catalog', {}).get('sha256') != DERIVED_SHA256
            or control.get('actual_private_library_path') != PRIVATE_LIBRARY_PATH):
        raise ValueError('Requires the private solution102 scalar-stage capture')
    return stages.split_capture(dict(payload, capture_variant=stages.VARIANT))


def archive_sources(directory, context, verifications):
    control = verify_private_catalog(os.environ)
    if len(verifications) != 2 or any(row['control'] != control for row in verifications):
        raise ValueError('Private catalog differs from bootstrap verification')
    context['source_sha256'].update(verify_sources())
    context[CONTROL_KEY] = control
    context['private_catalog_bootstrap_verifications'] = copy.deepcopy(verifications)
    target = Path(directory) / 'dw_solution102_private_catalog_manifest.json'
    data = Path(MANIFEST_PATH).read_bytes()
    if hashlib.sha256(data).hexdigest() != MANIFEST_SHA256:
        raise ValueError('Private catalog manifest changed before archival')
    with target.open('xb') as stream:
        stream.write(data)
    context['private_catalog_manifest_artifact'] = {'path': target.name, 'sha256': MANIFEST_SHA256, 'bytes': len(data)}
    return stages.mitigated.training._archive_sources(directory, context)


def diagnostic_worker(*args, **kwargs):
    verifications = []
    def checked_worker(options, environment):
        if len(verifications) >= 2:
            raise ValueError('Unexpected private catalog worker validation phase')
        control = validate_worker(options, environment)
        phase = ('before_bootstrap', 'after_bootstrap')[len(verifications)]
        verifications.append({'phase': phase, 'control': control})
    def probe(*probe_args, **probe_kwargs):
        return ProjectionProbe(*probe_args, catalog_bootstrap_verifications=verifications, **probe_kwargs)
    def archive(directory, context):
        return archive_sources(directory, context, verifications)
    # Reuse the exact worker: its two validation calls bracket environment
    # bootstrap, and the second occurs before any trainer-component import.
    return base._scoped(stages.mitigated.diagnostic_worker,
                        ProjectionProbe=probe, validate_worker=checked_worker,
                        _archive_sources=archive, VARIANT=VARIANT, LIMITS=LIMITS)(*args, **kwargs)


def main(argv=None):
    return base._scoped(base.frozen.main, __doc__=__doc__,
                        configure_environment=configure_environment,
                        diagnostic_worker=diagnostic_worker)(argv)


if __name__ == '__main__':
    main()
