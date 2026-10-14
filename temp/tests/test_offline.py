import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
import sys
import zlib
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from download_datasets import sha256, inspect_dataset, package_zip, main


def tiny_png():
    def chunk(name, data):
        return len(data).to_bytes(4,'big')+name+data+(zlib.crc32(name+data)&0xffffffff).to_bytes(4,'big')
    return (b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',1,1,8,2,0,0,0))+
            chunk(b'IDAT',zlib.compress(b'\x00\xff\x00\x00'))+chunk(b'IEND',b''))


class OfflineTests(unittest.TestCase):
    def test_hash_deterministic(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'demo'; p.write_bytes(b'abc')
            self.assertEqual(sha256(p),'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')

    def test_empty_is_not_success(self):
        with tempfile.TemporaryDirectory() as td:
            stats=inspect_dataset(td)
            self.assertEqual(stats['images_on_disk'],0)
            self.assertEqual(stats['files'],0)

    def test_nested_zip_image_detected(self):
        with tempfile.TemporaryDirectory() as td:
            with ZipFile(Path(td)/'imgs.zip','w') as z:
                z.writestr('images/example.png',tiny_png())
            stats=inspect_dataset(td)
            self.assertEqual(stats['images_on_disk'],0)
            self.assertEqual(stats['images_inside_archives'],1)
            self.assertEqual(stats['corrupt_archives'],[])

    def test_packaging_without_git_folder(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td)/'output'; (d/'datasets'/'ncugd_glass_crack').mkdir(parents=True)
            (d/'datasets'/'ncugd_glass_crack'/'sample.png').write_bytes(tiny_png())
            (d/'datasets'/'ncugd_glass_crack'/'.git').mkdir()
            (d/'datasets'/'ncugd_glass_crack'/'.git'/'secret').write_text('omit')
            result=package_zip(d,Path(td)/'final.zip',{'ncugd_glass_crack'})
            with ZipFile(result['zip']) as z:
                self.assertEqual(z.namelist(),['datasets/ncugd_glass_crack/sample.png'])

    def test_cli_audit_only_real_fixture(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td)/'out'/'datasets'/'ncugd_glass_crack'
            d.mkdir(parents=True)
            (d/'sample.png').write_bytes(tiny_png())
            with contextlib.redirect_stdout(io.StringIO()):
                status=main(['--audit-only','--only','ncugd_glass_crack','--output',str(d.parents[1])])
            self.assertEqual(status,0)
            rec=json.loads((d.parents[1]/'download_status.json').read_text())
            self.assertEqual(rec['dataset_count'],1)
            self.assertTrue((Path(td)/'glass_crack_datasets.zip').is_file())

    def test_empty_cli_does_not_fake_zip(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'out'
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result=main(['--audit-only','--only','ncugd_glass_crack','--output',str(out)])
            self.assertEqual(result,2)
            self.assertFalse((Path(td)/'glass_crack_datasets.zip').exists())

    def test_only_run_keeps_other_dataset_records(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'out'
            for ds in ('ncugd_glass_crack','agdd_aircraft_glass'):
                (out/'datasets'/ds).mkdir(parents=True)
                (out/'datasets'/ds/'sample.png').write_bytes(tiny_png())
            with contextlib.redirect_stdout(io.StringIO()):
                main(['--audit-only','--no-zip','--only','agdd_aircraft_glass','--output',str(out)])
                status=main(['--audit-only','--no-zip','--only','ncugd_glass_crack','--output',str(out)])
            self.assertEqual(status,0)
            rec=json.loads((out/'download_status.json').read_text())
            self.assertEqual([r['id'] for r in rec['records']],['ncugd_glass_crack','agdd_aircraft_glass'])
            hashed={line.split(',')[0] for line in (out/'files_sha256.csv').read_text().splitlines()[1:]}
            self.assertEqual(hashed,{'ncugd_glass_crack','agdd_aircraft_glass'})


class CleaningTests(unittest.TestCase):
    def test_source_key_strips_roboflow_suffix(self):
        from clean_datasets import source_key
        self.assertEqual(source_key('IMG_6777_jpeg_jpg.rf.ec5a913a35f13bec5b3'),'IMG_6777')
        self.assertEqual(source_key('facadea_cr001_v1_jpeg.rf.8d9245f6c5'),'facadea_cr001_v1')
        self.assertEqual(source_key('13353123230'),'13353123230')

    def test_fix_yolo_lines(self):
        from clean_datasets import fix_yolo_lines
        lines=['2 0.1 0.1 0.5 0.1 0.5 0.5',
               '2 0.1 0.1 0.5 0.1 0.5 0.5',
               '1 0.5 0.5 0.2 0.4',
               '0 0.1 0.1 0.2 0.2 0.3 0.3',
               '0 -0.1 0.0 1.2 0.0 1.2 1.0',
               '0 0.1 0.2']
        out,fixes=fix_yolo_lines(lines,{0:0,1:1,2:3})
        self.assertEqual(out[0],'3 0.100000 0.100000 0.500000 0.100000 0.500000 0.500000')
        self.assertEqual(out[1],'1 0.400000 0.300000 0.600000 0.300000 0.600000 0.700000 0.400000 0.700000')
        self.assertEqual(out[2],'0 0.000000 0.000000 1.000000 0.000000 1.000000 1.000000')
        self.assertEqual(len(out),3)
        self.assertEqual(fixes,{'duplicate_dropped':1,'box_to_polygon':1,'zero_area_dropped':1,
                                'clipped_to_image':1,'too_few_points_dropped':1})

if __name__ == '__main__':
    unittest.main()
