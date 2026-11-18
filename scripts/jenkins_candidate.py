"""Trusted Jenkins agent entry point; never load orchestration from candidate code."""
import argparse
import json
import os
import re
import shutil
from pathlib import Path
from backend.app.services.jenkins import STAGES, bundle_digest
from backend.app.services.validation import ValidationService
from backend.app.services.ci_tools import redact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', *STAGES, 'handoff'])
    args = parser.parse_args()
    bundle_id = os.environ['BUNDLE_ID']
    if not re.fullmatch(r'[a-f0-9-]{36}', bundle_id):
        raise ValueError('Invalid bundle identifier')
    shared = Path(os.environ['CI_SHARED_ROOT']).resolve()
    bundle = shared / bundle_id
    workspace = Path(os.environ['WORKSPACE']).resolve()
    state_path = workspace / 'ci-result.json'
    if args.stage == 'prepare':
        manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
        if bundle_digest(bundle) != os.environ['BUNDLE_SHA256'] or manifest['bundle_sha256'] != os.environ['BUNDLE_SHA256']:
            raise ValueError('Source bundle digest mismatch')
        # Workspace is a fresh directory allocated by Jenkins for this build.
        for name in ('candidate', 'baseline'):
            shutil.copytree(bundle / name, workspace / name)
        if bundle_digest(workspace) != manifest['bundle_sha256']:
            raise ValueError('Copied source digest mismatch')
        manifest['checks'] = {}
        state_path.write_text(json.dumps(manifest), encoding='utf-8')
        return
    data = json.loads(state_path.read_text(encoding='utf-8'))
    if args.stage == 'handoff':
        if set(data['checks']) != set(STAGES):
            raise ValueError('Missing validation stages')
        data['handoff'] = 'BACKEND_REANALYSIS_REQUIRED'
    else:
        # Prevent recursively submitting another Jenkins build.
        os.environ['CI_EXECUTOR'] = 'local'
        service = ValidationService()
        result = service.validate(workspace / 'candidate', requested=[args.stage], generated_test_files=data['regression'], generated_integration_files=data['targeted_integration'], variant='Candidate', baseline_root=workspace / 'baseline', ci_context={k: data[k] for k in ('analysis_id','candidate_id','attempt')})
        data['checks'][args.stage] = result.checks[args.stage]
    state_path.write_text(redact(json.dumps(data)), encoding='utf-8')


if __name__ == '__main__':
    main()
