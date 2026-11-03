"""Read-only NEFIS 5 split-file reader for Delft3D map outputs.

The disk layout is documented by Deltares' LGPL Delft3D MATLAB routines
vs_use.m and vs_let.m. This reader implements only one-dimensional groups
(the map record/time axis) and reads one element/record at a time, so a
multi-GB running simulation need not be loaded into memory.

Format/algorithm reference: Deltares Delft3D MATLAB vs_use.m and vs_let.m.
Copyright (C) 2011-2020 Stichting Deltares.
https://github.com/Deltares/Delft3D

Reimplemented in Python on 2026-11-01 for the scope described above; this is
not the original Deltares software. See NOTICE.txt and LICENSE-LGPL-2.1.txt.
Distributed WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

SPDX-License-Identifier: LGPL-2.1-or-later
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import struct


@dataclass(frozen=True)
class Element:
    name: str
    type: str
    bytes: int
    itemsize: int
    quantity: str
    unit: str
    description: str
    dimensions: tuple[int, ...]  # on-disk fastest dimension first

    @property
    def shape(self):
        """NumPy shape in C order, reversed from NEFIS/Fortran order."""
        return self.dimensions[::-1]


class NefisReader:
    """Read existing .dat/.def without DLLs. Indices are Python zero-based.

    ``reader.read('map-sed-series', 'LYRFRAC', -1)`` reads the last record.
    Returned array dimensions are reversed NEFIS dimensions, including all
    singleton dimensions. For this user's map data the trailing dimensions
    are (M, N); do not squeeze a sediment-fraction axis blindly.
    """
    def __init__(self, dat_path, def_path=None):
        self.dat_path = Path(dat_path)
        self.def_path = Path(def_path) if def_path else self.dat_path.with_suffix('.def')
        self._dat = self.dat_path.open('rb')
        self._def = None
        self.elements = {}
        self.groups = {}
        try:
            self._def = self.def_path.open('rb')
            hdr = self._read_at(self._dat, 0, 60)
            hdr_def = self._read_at(self._def, 0, 60)
            if b'NEFIS' not in hdr or b'File; 5.' not in hdr:
                raise ValueError('Only NEFIS 5 split files are supported; this is not a NEFIS 5 data file.')
            if b'NEFIS' not in hdr_def or b'File; 5.' not in hdr_def:
                raise ValueError('The definition file is not NEFIS 5.')
            if hdr[-1:] not in (b'L', b'B') or hdr_def[-1:] != hdr[-1:]:
                raise ValueError('Unsupported or inconsistent byte ordering.')
            self.endian = '<' if hdr[-1:] == b'L' else '>'
            self.nil = 2**64 - 1
            self._scan()
        except Exception:
            self.close()
            raise

    @staticmethod
    def _read_at(fh, offset, n):
        if offset < 0:
            raise ValueError('Negative NEFIS offset.')
        fh.seek(offset)
        value = fh.read(n)
        if len(value) != n:
            raise EOFError(f'Incomplete NEFIS data at {offset}; requested {n} bytes, found {len(value)}. The simulation may still be writing.')
        return value

    def _unpack_at(self, fh, offset, fmt):
        fmt = self.endian + fmt
        return struct.unpack(fmt, self._read_at(fh, offset, struct.calcsize(fmt)))

    @staticmethod
    def _text(raw):
        return raw.decode('latin1').rstrip(' \x00')

    def _records(self, fh, table_offset, expected):
        pointers = self._unpack_at(fh, table_offset, '997Q')
        seen = set()
        size = fh.seek(0, 2)
        for first in pointers:
            pointer = first
            while pointer != self.nil:
                if pointer in seen:
                    raise ValueError('Cyclic or repeated NEFIS hash-chain pointer.')
                seen.add(pointer)
                if not 68 + 997 * 8 <= pointer < size:
                    raise ValueError(f'Invalid NEFIS hash pointer {pointer}.')
                raw = self._read_at(fh, pointer, 24)
                link, length = struct.unpack(self.endian + '2Q', raw[:16])
                kind = self._text(raw[16:24]).strip()
                if kind not in expected:
                    raise ValueError(f'Unexpected NEFIS record type {kind!r}.')
                yield pointer, length, kind
                pointer = link

    def _scan(self):
        for offset, length, kind in self._records(self._def, 68, {'1'}):
            raw = self._read_at(self._def, offset + 24, 156)
            name = self._text(raw[:16])
            typ = self._text(raw[16:24])
            nbytes, itemsize = struct.unpack(self.endian + 'QI', raw[24:36])
            ndim, *dims = struct.unpack(self.endian + '6I', raw[132:156])
            dims = tuple(dims[:max(1, ndim)])
            if not 0 <= ndim <= 5 or any(n <= 0 for n in dims):
                raise ValueError(f'Invalid dimensions for {name}.')
            if math.prod(dims) * itemsize != nbytes:
                raise ValueError(f'Element byte size does not match dimensions: {name}.')
            self.elements[name] = Element(name, typ, nbytes, itemsize,
                self._text(raw[36:52]), self._text(raw[52:68]),
                self._text(raw[68:132]), dims)
        cells = {}
        for offset, length, kind in self._records(self._def, 68 + 997 * 8, {'2'}):
            raw = self._read_at(self._def, offset + 24, 28)
            name = self._text(raw[:16])
            nbytes, nelms = struct.unpack(self.endian + 'QI', raw[16:28])
            if nelms > 10000:
                raise ValueError('Unreasonable NEFIS element count.')
            names = self._read_at(self._def, offset + 52, nelms * 16)
            names = [self._text(names[i:i+16]) for i in range(0, len(names), 16)]
            element_bytes = sum(self.elements[e].bytes for e in names)
            # NEFIS may round the stored cell size up to a four-byte boundary.
            if not element_bytes <= nbytes <= element_bytes + 3:
                raise ValueError(f'Cell size mismatch for {name}.')
            cells[name] = {'bytes': nbytes, 'elements': names}
        definitions = {}
        for offset, length, kind in self._records(self._def, 68 + 2 * 997 * 8, {'3'}):
            raw = self._read_at(self._def, offset + 24, 76)
            name, cell = self._text(raw[:16]), self._text(raw[16:32])
            ndim, *dims_orders = struct.unpack(self.endian + '11I', raw[32:])
            if ndim != 1:
                raise NotImplementedError('Only one-dimensional map output groups are supported.')
            if dims_orders[5] != 1:
                raise ValueError('Unsupported NEFIS group dimension order.')
            definitions[name] = {'cell': cell, 'size': dims_orders[0]}
        for offset, length, kind in self._records(self._dat, 68, {'4', '5'}):
            raw = self._read_at(self._dat, offset + 24, 32)
            name, defname = self._text(raw[:16]), self._text(raw[16:32])
            definition = definitions[defname]
            cell = cells[definition['cell']]
            variable = kind == '5'
            count = self._max_index(offset + 424) if variable else definition['size']
            if variable != (definition['size'] == 0):
                raise ValueError('Variable dimension mismatch.')
            element_offsets = {}
            eoff = 0
            for element in cell['elements']:
                element_offsets[element] = eoff
                eoff += self.elements[element].bytes
            self.groups[name] = {'offset': offset, 'variable': variable,
                'count': count, 'cell_bytes': cell['bytes'],
                'elements': cell['elements'], 'element_offsets': element_offsets}

    def _max_index(self, pointer):
        index = 0
        for shift in (24, 16, 8, 0):
            table = self._unpack_at(self._dat, pointer, '256Q')
            valid = [i for i, p in enumerate(table) if p != self.nil]
            if not valid:
                return 0
            last = max(valid)
            index += last << shift
            pointer = table[last]
        return index  # stored indices are one-based, max equals record count

    def _record_offset(self, group, index):
        if index < 0:
            index += group['count']
        if not 0 <= index < group['count']:
            raise IndexError(f'Record {index} outside [0, {group["count"]}).')
        if not group['variable']:
            return group['offset'] + 416 + index * group['cell_bytes']
        stored_index = index + 1
        pointer = group['offset'] + 424
        for shift in (24, 16, 8, 0):
            pointer = self._unpack_at(self._dat, pointer + ((stored_index >> shift) & 255) * 8, 'Q')[0]
            if pointer == self.nil:
                raise IndexError(f'Missing NEFIS record {index}.')
        return pointer

    def raw(self, group_name, element_name, record=0):
        """Read one complete element as bytes (no NumPy dependency)."""
        group = self.groups[group_name]
        offset = group['element_offsets'][element_name]
        return self._read_at(self._dat, self._record_offset(group, record) + offset,
                             self.elements[element_name].bytes)

    def read(self, group_name, element_name, record=0):
        """Read one record as ndarray with element.shape, C/NumPy ordering."""
        import numpy as np
        elm = self.elements[element_name]
        types = {'REAL': 'f', 'INTEGER': 'i', 'LOGICAL': 'i', 'COMPLEX': 'c'}
        if elm.type == 'CHARACTE':
            dtype = f'S{elm.itemsize}'
        else:
            dtype = f'{self.endian}{types[elm.type]}{elm.itemsize}'
        return np.frombuffer(self.raw(group_name, element_name, record), dtype=dtype).reshape(elm.shape).copy()

    def latest_complete_record(self, *group_names):
        """Last candidate record whose entire cell extents exist in every group.

        This is an on-disk bounds check, not an atomic writer commit check.
        A preallocated cell may exist before the writer finishes filling it.
        For a live simulation compare ITMAPC with ITMAPS, validate requested
        arrays, and compare a second read before accepting a snapshot. For a
        strict snapshot use output after the run has closed/flushed its files.
        Group metadata/counts form an opening-time snapshot; reopen to refresh.
        """
        if not group_names:
            raise ValueError('At least one group is required.')
        groups = [self.groups[name] for name in group_names]
        size = self._dat.seek(0, 2)
        for index in range(min(g['count'] for g in groups) - 1, -1, -1):
            try:
                complete = all(self._record_offset(g, index) + g['cell_bytes'] <= size
                               for g in groups)
            except (IndexError, EOFError):
                complete = False
            if complete:
                return index
        raise ValueError('No complete common record exists in the requested groups.')

    def scalar(self, group, element, record=0):
        """Read a scalar numeric element without NumPy."""
        e = self.elements[element]
        if math.prod(e.dimensions) != 1:
            raise ValueError('Element is not scalar.')
        codes = {('INTEGER',4):'i', ('INTEGER',8):'q', ('REAL',4):'f', ('REAL',8):'d'}
        return struct.unpack(self.endian+codes[(e.type,e.itemsize)], self.raw(group,element,record))[0]

    def metadata(self):
        return {'dat_path': str(self.dat_path), 'endian': self.endian,
                'elements': {k: asdict(v) for k, v in self.elements.items()},
                'groups': self.groups}

    def close(self):
        for fh in (getattr(self, '_dat', None), getattr(self, '_def', None)):
            if fh is not None:
                fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == '__main__':
    import json
    import sys
    with NefisReader(sys.argv[1]) as r:
        print(json.dumps(r.metadata(), indent=2))
