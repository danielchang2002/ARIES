from utils import *
from msa_tools import *

def run_clustalw(msa, msa_name, clean=False):
    infile = f'{temp_dir}/{msa_name}.fasta'
    oufile = f'{temp_dir}/{msa_name}_clustalw.aln'
    dnd_src = f'{temp_dir}/{msa_name}.dnd'
    dnd_dst = f'{temp_dir}/{msa_name}_clustalw.dnd'
    spawn_fasta(msa, infile)
    # Run ClustalW (assumes 'clustalw2' or 'clustalw' is in PATH)
    clustalw_cmd = [
        "clustalw2",
        f"-INFILE={infile}",
        '-TYPE=PROTEIN',
        '-MATRIX=BLOSUM',
        "-OUTPUT=FASTA",
        "-SEED=123",
        f"-OUTFILE={oufile}"
    ]
    log = subprocess.run(clustalw_cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if os.path.exists(dnd_src) and not os.path.exists(dnd_dst):
        try:
            os.replace(dnd_src, dnd_dst)
        except Exception:
            pass
    if clean:
        os.system(f'rm -r {infile}')
        os.system(f'rm -r {dnd_dst}')
    return oufile

def run_clustalo(msa, msa_name, clean=False):
    infile = f'{temp_dir}/{msa_name}.fasta'
    oufile = f'{temp_dir}/{msa_name}_clustalo.aln'
    dndfile = f'{temp_dir}/{msa_name}_clustalo.dnd'
    spawn_fasta(msa, infile)
    clustalo_cmd = [
        "clustalo",
        "-i", infile,
        "-o", oufile,
        "--outfmt", "fasta",
        "--force",
        "--threads", '10',
        "--verbose",
        "--guidetree-out", dndfile
    ]
    log = subprocess.run(clustalo_cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if clean:
        os.system(f'rm -r {infile}')
        os.system(f'rm -r {temp_dir}/{msa_name}.dnd')
    return oufile

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
        
