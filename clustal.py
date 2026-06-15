from utils import *
from msa_tools import *


def _newick_complete(path):
    """Returns True once the newick file is fully flushed (ends with ';')."""
    try:
        with open(path, 'rb') as f:
            data = f.read().rstrip()
        return data.endswith(b';')
    except OSError:
        return False


def _kill_after_dnd(proc, dnd_path):
    """Poll until dnd_path is completely written, then terminate the process."""
    while True:
        if proc.poll() is not None:
            break
        if _newick_complete(dnd_path):
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            break
        time.sleep(0.01)


def run_clustalw(msa, msa_name, clean=False, guidetree_only=False):
    infile  = f'{temp_dir}/{msa_name}.fasta'
    oufile  = f'{temp_dir}/{msa_name}_clustalw.aln'
    dnd_src = f'{temp_dir}/{msa_name}.dnd'
    dnd_dst = f'{temp_dir}/{msa_name}_clustalw.dnd'
    spawn_fasta(msa, infile)
    clustalw_cmd = [
        "clustalw2",
        f"-INFILE={infile}",
        '-TYPE=PROTEIN',
        '-MATRIX=BLOSUM',
        "-OUTPUT=FASTA",
        "-SEED=123",
        f"-OUTFILE={oufile}",
    ]
    if guidetree_only:
        proc = subprocess.Popen(clustalw_cmd,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _kill_after_dnd(proc, dnd_src)
    else:
        subprocess.run(clustalw_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(dnd_src) and not os.path.exists(dnd_dst):
        try:
            os.replace(dnd_src, dnd_dst)
        except Exception:
            pass
    if clean:
        os.system(f'rm -r {infile}')
        os.system(f'rm -r {dnd_dst}')
    return dnd_dst if guidetree_only else oufile


def run_clustalo(msa, msa_name, clean=False, guidetree_only=False):
    infile  = f'{temp_dir}/{msa_name}.fasta'
    oufile  = f'{temp_dir}/{msa_name}_clustalo.aln'
    dndfile = f'{temp_dir}/{msa_name}_clustalo.dnd'
    spawn_fasta(msa, infile)
    if guidetree_only:
        clustalo_cmd = [
            "clustalo",
            "-i", infile,
            "--guidetree-out", dndfile,
            "--threads", '10',
            "--force",
        ]
        proc = subprocess.Popen(clustalo_cmd,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _kill_after_dnd(proc, dndfile)
    else:
        clustalo_cmd = [
            "clustalo",
            "-i", infile,
            "-o", oufile,
            "--outfmt", "fasta",
            "--force",
            "--threads", '10',
            "--verbose",
            "--guidetree-out", dndfile,
        ]
        subprocess.run(clustalo_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # clustalo skips guide tree generation for <3 sequences; write a trivial newick tree.
    if not os.path.exists(dndfile):
        n = len(msa)
        if n == 1:
            newick = '(seq0:0.0);'
        elif n == 2:
            newick = '(seq0:0.5,seq1:0.5);'
        else:
            newick = '(' + ','.join(f'seq{i}:1.0' for i in range(n)) + ');'
        with open(dndfile, 'w') as f:
            f.write(newick + '\n')
    if clean:
        os.system(f'rm -r {infile}')
        os.system(f'rm -r {temp_dir}/{msa_name}.dnd')
    return dndfile if guidetree_only else oufile

class MSAClustalO:
    def __init__(self, **kwargs):
        self.clustal = lambda seqs, aln_name: run_clustalo(seqs, aln_name)
    
    def align(self, seqs, **kwargs):
        aln_name = kwargs['aln_name']
        self.clustal(seqs, aln_name)
        alns = SeqIO.to_dict(SeqIO.parse(f'{temp_dir}/{aln_name}_clustalo.aln', "fasta"))
        alns = [str(alns[f'seq{i}'].seq) for i in range(len(seqs))]
        return alns

class MSAClustalW:
    def __init__(self, **kwargs):
        self.clustal = lambda seqs, aln_name: run_clustalw(seqs, aln_name)
    
    def align(self, seqs, **kwargs):
        aln_name = kwargs['aln_name']
        self.clustal(seqs, aln_name)
        alns = SeqIO.to_dict(SeqIO.parse(f'{temp_dir}/{aln_name}_clustalw.aln', "fasta"))
        alns = [str(alns[f'seq{i}'].seq) for i in range(len(seqs))]
        return alns
        
