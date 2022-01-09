from pyvolve import read_tree, Model, Partition, Evolver

def build_star_tree(b_lengths):
    bls = b_lengths.split(",")

    newick = "("
    for i, bl in enumerate(bls):
        newick += "t{}:{}".format(i+1, bl)
        if i<len(bls)-1: newick += ","
    newick += ");"

    return newick

def evolve_seqs_full_homogeneity(
        fasta_file=None, 
        nb_sites=100, 
        branch_lengths="0.1,0.1", 
        mu=None,
        state_freqs=None,
        verbose=False):

    # Evolve sequences with complete homogeneity
    # (all sites and branches evolve according to a single model)
    if verbose: print("Evolving new sequences with the amazing Pyvolve for {}".format(fasta_file))

    tree = read_tree(tree=build_star_tree(branch_lengths))

    if mu is not None and state_freqs is not None :  
        m = Model("nucleotide", {"mu":mu, "state_freqs":state_freqs})
    else:
        m = Model("nucleotide")

    p = Partition(size=nb_sites, models=m)
    e = Evolver(partitions=p, tree=tree)
    e(seqfile=fasta_file, infofile=False, ratefile=False)

    seqdict = e.get_sequences(anc=True)
    
    return seqdict["root"], [seqdict[s] for s in seqdict if s != "root"]
