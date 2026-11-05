"""Official RRWNet evaluation code and data, pinned to the commit used in the paper.

- Code: https://github.com/j-morano/rrwnet (MIT License) at commit 03069bb. Its ``eval/test_utils.py`` (A/V classification)
  and ``eval/topo_metric.py`` (Inf/Corr connectivity, from Araújo et al., MICCAI 2019) are used without changes.
- Data: ``_Evaluation_Data.zip`` of the RRWNet release, from which only the references and FOV masks of RITE (gt_hu), HRF
  (gt_chen) and LES-AV (gt_orlando) are extracted.
- RRWNet baseline (optional): the official weights trained on RITE and on HRF and ``_Data.zip``, which holds the official
  preprocessed inputs of RITE and HRF.
"""
from pathlib import Path
import subprocess
import urllib.request
import zipfile

OFFICIAL_COMMIT = '03069bb673f87d4b3cee37df378e0f15e5ade280'
REPOSITORY = 'https://github.com/j-morano/rrwnet.git'
RELEASE = 'https://github.com/j-morano/rrwnet/releases/download'
EVALUATION_ARCHIVE = ('preds-n-data', '_Evaluation_Data.zip', 555294199)
BASELINE_FILES = [('weights', 'rrwnet_RITE_1.pth', 248304547), ('weights', 'rrwnet_HRF_0.pth', 248304547),
                  ('preds-n-data', '_Data.zip', 69284042)]
REFERENCE_FOLDERS = ('RITE-test/gt_hu', 'RITE-test/masks', 'HRF/gt_chen', 'HRF/masks', 'LES-AV/gt_orlando', 'LES-AV/masks')
OFFICIAL_INPUT_FOLDERS = ('RITE/test/enhanced', 'RITE/test/enhanced_masks',
                          'HRF_AVLabel_191219/test_karlsson_w1024/enhanced',
                          'HRF_AVLabel_191219/test_karlsson_w1024/enhanced_masks')


def _download(tag, name, size, folder):
    dest = Path(folder) / name
    if dest.exists() and dest.stat().st_size == size:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.part')
    print(f'Downloading {name} ({size / 1e6:.0f} MB)', flush=True)
    urllib.request.urlretrieve(f'{RELEASE}/{tag}/{name}', tmp)
    if tmp.stat().st_size != size:
        raise RuntimeError(f'Incomplete download: {name}')
    tmp.replace(dest)
    return dest


def _extract(archive, folders, target, prefix=''):
    target = Path(target)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            relative = name.removeprefix(prefix)
            if name.startswith('__MACOSX/') or name.endswith('/') or str(Path(relative).parent) not in folders:
                continue
            if Path(relative).is_absolute() or '..' in Path(relative).parts:
                raise ValueError('Unsafe archive path')
            path = target / relative
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(z.read(name))


def ensure_assets(root, baseline=False):
    """Prepare ``root`` with the pinned RRWNet code, the official references/FOVs and, if ``baseline``, the RRWNet weights
    and official inputs. Returns a dict of paths."""
    root = Path(root)
    repo = root / 'rrwnet'
    if not repo.exists():
        subprocess.run(['git', 'clone', '--quiet', REPOSITORY, str(repo)], check=True)
        subprocess.run(['git', '-C', str(repo), 'checkout', '--quiet', OFFICIAL_COMMIT], check=True)
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError(f'{repo} is at {commit}; expected {OFFICIAL_COMMIT}')
    references = root / 'references'
    if not (references / 'complete').exists():
        archive = _download(*EVALUATION_ARCHIVE, root / 'downloads')
        _extract(archive, REFERENCE_FOLDERS, references, prefix='_Evaluation_Data/')
        (references / 'complete').write_text(OFFICIAL_COMMIT)
    paths = dict(repo=repo, references=references)
    if baseline:
        files = {name: _download(tag, name, size, root / 'downloads') for tag, name, size in BASELINE_FILES}
        inputs = root / 'official_inputs'
        if not (inputs / 'complete').exists():
            _extract(files['_Data.zip'], OFFICIAL_INPUT_FOLDERS, inputs)
            (inputs / 'complete').write_text(OFFICIAL_COMMIT)
        paths.update(weights_rite=files['rrwnet_RITE_1.pth'], weights_hrf=files['rrwnet_HRF_0.pth'],
                     official_inputs=inputs)
    return paths
