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
        nwk_tree,
        nb_sites=10, 
        fasta_file=False,
        subst_rates=None,
        state_freqs=None,
        return_anc=True,
        seed=None,
        verbose=False):

    # Evolve sequences
    if verbose: print("Evolving new sequences with the amazing "\
            "Pyvolve for {}".format(fasta_file))
    tree = read_tree(tree=nwk_tree)

    parameters = None
    if subst_rates is not None or state_freqs is not None:
        parameters = dict()

        if subst_rates is not None:
            parameters.update(mu=subst_rates)

        if state_freqs is not None:
            parameters.update(state_freqs=state_freqs)

    m = Model("nucleotide", parameters=parameters) 

    p = Partition(size=nb_sites, models=m)
    e = Evolver(partitions=p, tree=tree)

    e(seqfile=fasta_file, infofile=False, ratefile=False,
            seed=seed)

    seqdict = e.get_sequences(anc=return_anc)
    
    return seqdict["root"], [seqdict[s] for s in seqdict if s != "root"]
