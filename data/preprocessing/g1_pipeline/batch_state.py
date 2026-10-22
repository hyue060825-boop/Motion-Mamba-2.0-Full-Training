"""Persistent per-clip work ownership and atomic artifact commits for full batch."""
import fcntl
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from .common import digest,sha,now

TERMINAL={'QA_PASS','REJECTED','ERROR','UNSUPPORTED','MISSING'}


def item_key(row,semantic_version):
    return digest({'release':row['dataset_release'],'source':row['source_motion_id'],
        'interval':row.get('native_interval') or [row.get('native_start_sec'),row.get('native_end_sec')],
        'mirror':row['mirror_flag'],'semantic_version':semantic_version})


class State:
    def __init__(self,directory,fingerprint):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        (self.directory/'locks').mkdir(exist_ok=True);self.connection=sqlite3.connect(self.directory/'state.sqlite',timeout=60,isolation_level=None)
        self.connection.row_factory=sqlite3.Row;self.connection.execute('PRAGMA journal_mode=WAL');self.connection.execute('PRAGMA synchronous=FULL')
        self.connection.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
        self.connection.execute('CREATE TABLE IF NOT EXISTS items (sample_id TEXT PRIMARY KEY,item_key TEXT UNIQUE NOT NULL,input_json TEXT NOT NULL,status TEXT NOT NULL,attempt INTEGER NOT NULL DEFAULT 0,worker_pid INTEGER,heartbeat TEXT,output_json TEXT,error TEXT)')
        expected=json.dumps(fingerprint,sort_keys=True);self.connection.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)',('fingerprint',expected))
        actual=self.connection.execute('SELECT value FROM metadata WHERE key=?',('fingerprint',)).fetchone()[0]
        if actual!=expected:raise ValueError('BATCH_NAMESPACE_FINGERPRINT_CHANGED')

    def register(self,rows,semantic_version):
        with self.transaction():
            for row in rows:
                key=item_key(row,semantic_version);encoded=json.dumps(row,sort_keys=True)
                old=self.connection.execute('SELECT item_key,input_json FROM items WHERE sample_id=?',(row['sample_id'],)).fetchone()
                if old:
                    if old['item_key']!=key or old['input_json']!=encoded:raise ValueError('REGISTERED_BATCH_INPUT_CHANGED')
                    continue
                self.connection.execute('INSERT INTO items (sample_id,item_key,input_json,status) VALUES (?,?,?,?)',(row['sample_id'],key,encoded,'DISCOVERED'))

    @contextmanager
    def transaction(self):
        self.connection.execute('BEGIN IMMEDIATE')
        try:yield;self.connection.execute('COMMIT')
        except BaseException:self.connection.execute('ROLLBACK');raise

    @contextmanager
    def own(self,sample_id,*,resume=False,retry_failed=False):
        record=self.connection.execute('SELECT * FROM items WHERE sample_id=?',(sample_id,)).fetchone()
        if record is None:raise KeyError(sample_id)
        lock=(self.directory/'locks'/(record['item_key']+'.lock')).open('a+')
        try:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:yield None;return
            record=dict(self.connection.execute('SELECT * FROM items WHERE sample_id=?',(sample_id,)).fetchone())
            if record['status'] in TERMINAL:
                valid=self.verify(record)
                if valid and (record['status']!='ERROR' or not retry_failed):yield None;return
                if not resume:raise ValueError('BATCH_EXISTING_OUTPUT_REQUIRES_RESUME')
            elif record['attempt'] and not resume:raise ValueError('BATCH_INTERRUPTED_ITEM_REQUIRES_RESUME')
            with self.transaction():
                self.connection.execute('UPDATE items SET status=?,attempt=attempt+1,worker_pid=?,heartbeat=?,error=NULL WHERE sample_id=?',('VALIDATED',os.getpid(),now(),sample_id))
            yield json.loads(record['input_json'])
        finally:lock.close()

    def heartbeat(self,sample_id,status):
        if status not in {'VALIDATED','RESAMPLED','RETARGETED','EXPORTED','ROUNDTRIP_PASS'}:raise ValueError('INVALID_INTERMEDIATE_STATE')
        changed=self.connection.execute('UPDATE items SET status=?,heartbeat=? WHERE sample_id=? AND worker_pid=?',(status,now(),sample_id,os.getpid())).rowcount
        if changed!=1:raise ValueError('BATCH_WORKER_DOES_NOT_OWN_ITEM')

    @staticmethod
    def verify(record):
        if record['status'] not in TERMINAL or not record['output_json']:return False
        output=json.loads(record['output_json'])
        bindings=output.get('artifact_hashes',{})
        if not bindings:return False
        for path,bound in bindings.items():
            if not Path(path).is_file() or sha(path)!=bound:return False
        if record['status']=='QA_PASS':
            import numpy as np
            contract=json.loads(Path(output['contract_path']).read_text())
            if contract.get('status')!='PASS':return False
            q=np.load(output['qpos_path'],mmap_mode='r');features=np.load(output['features_path'],mmap_mode='r');n=output['frames']
            if q.shape!=(n,36) or features.shape!=(n,414) or not np.isfinite(q).all() or not np.isfinite(features).all():return False
        return True

    def commit(self,sample_id,status,output,error=None):
        if status not in TERMINAL:raise ValueError('INVALID_TERMINAL_STATE')
        record={'status':status,'output_json':json.dumps(output,sort_keys=True,allow_nan=False)}
        if not self.verify(record):raise ValueError('BATCH_ARTIFACT_COMMIT_REJECTED')
        with self.transaction():
            changed=self.connection.execute('UPDATE items SET status=?,output_json=?,error=?,heartbeat=?,worker_pid=NULL WHERE sample_id=? AND worker_pid=?',
                (status,record['output_json'],error,now(),sample_id,os.getpid())).rowcount
            if changed!=1:raise ValueError('BATCH_COMMIT_OWNER_MISMATCH')

    def rows(self):return [dict(r) for r in self.connection.execute('SELECT * FROM items ORDER BY sample_id')]
    def close(self):self.connection.close()
