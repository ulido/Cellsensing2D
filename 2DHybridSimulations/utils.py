import json
import gzip
from multiprocess import Pool
from tqdm.auto import tqdm

def json_cache_write(name, data):
    with gzip.open(name + '.json.gz', 'wt', encoding='utf8') as f:
        json.dump(data, f)
        
def json_cache_read(name):
    with gzip.open(name + '.json.gz', 'rt', encoding='utf8') as f:
        return json.load(f)
    
def send_parallel_jobs(name, function, todo):
    try:
        return json_cache_read(name)
    except (IOError, ValueError):
        pass

    with Pool() as p:
        results = [r for r in tqdm(p.imap_unordered(function, todo), total=len(todo), smoothing=0)]
    
    json_cache_write(name, results)
    return results