"""Preserve an official profile until the new Skeleton calibration is validated."""

import math
from pathlib import Path
import shutil
import tempfile

import yaml


class CalibrationRejected(ValueError):
    pass


class CalibrationGuard:
    parts = ('captures', 'calibrations')

    def __init__(self, home):
        self.home = Path(home).resolve()
        self.profile = self.home / 'glove'
        self.pending = self.home / '.pending_dex_calibration'

    def begin(self):
        if self.pending.exists():
            return  # A retry in this session must retain the original checkpoint.
        self.home.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.prepare_calibration_', dir=self.home))
        for part in self.parts:
            active = self.profile / part
            if active.is_dir():
                self._check_path(active)
                shutil.copytree(active, staging / part)
        staging.rename(self.pending)

    def validate(self, result_ids):
        """Read the numerically valid Skeleton scale before preview."""
        remaining = set(result_ids)
        if len(remaining) != 1:
            raise CalibrationRejected('SDK 未提供唯一的 Skeleton 标定结果 ID')
        for path in (self.profile / 'calibrations').glob('*/record.yaml'):
            self._check_path(path)
            record = yaml.safe_load(path.read_text(encoding='utf-8'))
            record_id = record['_metadata']['record_id']
            if record_id not in remaining:
                continue
            try:
                raw = float(record['data']['raw_scale'])
                scale = float(record['data']['scale'])
            except (KeyError, TypeError, ValueError) as exc:
                raise CalibrationRejected('新标定缺少可读取的骨骼缩放系数') from exc
            if not all(math.isfinite(value) and value > 0 for value in (raw, scale)):
                raise CalibrationRejected(f'新标定缩放异常 (raw_scale={raw:.6g}, scale={scale:.6g})')
            remaining.remove(record_id)
        if remaining:
            raise CalibrationRejected('找不到与此次 SDK 结果匹配的标定记录')
        return scale

    def commit(self):
        """Keep the new record after the wearer confirms the preview."""
        if self.pending.exists():
            archive = Path(tempfile.mkdtemp(prefix='previous_calibration_', dir=self.home))
            self._check_path(self.pending)
            self.pending.rename(archive / 'previous')

    def rollback(self):
        """Restore the checkpoint only when the SDK has released its store."""
        if not self.pending.exists():
            return
        self._check_path(self.pending)
        archive = Path(tempfile.mkdtemp(prefix='rejected_calibration_', dir=self.home))
        self.profile.mkdir(exist_ok=True)
        for part in self.parts:
            active = self.profile / part
            previous = self.pending / part
            if active.exists():
                self._check_path(active)
                active.rename(archive / part)
            if previous.exists():
                self._check_path(previous)
                shutil.copytree(previous, active)
        self.pending.rename(archive / 'previous')

    def _check_path(self, path):
        if not path.resolve(strict=True).is_relative_to(self.home):
            raise ValueError(f'标定目录超出数据目录：{path}')
