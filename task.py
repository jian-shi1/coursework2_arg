import msprime
import tsinfer
import tskit
from Bio import bgzf
import numpy as np
from scipy.optimize import minimize
import builtins
import sys
import os
from shutil import rmtree
import pathlib
import json
import subprocess
import zarr
import concurrent.futures

k_reference = 50
k_learning = 13
k_validation = 3
m = 1
ploidy = 2
smp = {"A": k_reference + k_learning, "B": k_reference + k_learning, "C": k_validation}
alphas = [0.5]
dem_size = 1e4
T_big_split = 1000
T_C = 100
sequence_len=46709983
recomb_rate=1.72443e-08
replicates=24
mut_rate = 1.29e-08
types = ("0", "1", "2", "3", "4")
methods = ['COBYLA', 'COBYQA', 'SLSQP', 'trust-constr', 'Nelder-Mead', 'L-BFGS-B', 'Powell', 'TNC']
bnd = ((0.0, 1.0),)
start_guess = np.array([0.5])
vecsize = k_reference*ploidy+1
long_vecsize = 2*k_reference*ploidy+1
python = sys.executable

# def build_dem(alpha):
dem = msprime.Demography()
dem.add_population(name="A", initial_size=dem_size)
dem.add_population(name="B", initial_size=dem_size)
dem.add_population(name="P0", initial_size=dem_size/10) 
dem.add_population(name="C", initial_size=dem_size/10)
dem.add_admixture(time=T_C, derived="C", ancestral=["A", "B"], proportions=[0.5, 0.5])
dem.add_population_split(time=T_big_split, derived=["A", "B"], ancestral="P0")
# return dem

def natural_arg_sets(ts):
    popA = ts.samples(population_id=0)
    popB = ts.samples(population_id=1)
    refA = popA[:(k_reference*ploidy)]
    refB = popB[:(k_reference*ploidy)]
    # reference = np.concat([refA, refB])
    reference = np.zeros(2 * k_reference * ploidy)
    reference[:k_reference * ploidy] = refA
    reference[k_reference * ploidy:] = refB
    learnA = popA[(k_reference*ploidy) : (k_reference + k_learning)*ploidy]
    learnB = popB[(k_reference*ploidy): (k_reference + k_learning)*ploidy]
    validation_set = ts.samples(population_id=3)
    return reference, learnA, learnB, validation_set

def infered_arg_sets(ts):
    refA = ts.samples(population_id=0)
    refB = ts.samples(population_id=2)
    reference = np.zeros(2 * k_reference * ploidy)
    reference[:k_reference * ploidy] = refA
    reference[k_reference * ploidy:] = refB
    learnA = ts.samples(population_id=1)
    learnB = ts.samples(population_id=3)
    validation_set = ts.samples(population_id=4)
    return reference, learnA, learnB, validation_set

def get_sub_ARG(treeseq, reference, sample):
    return treeseq.simplify(np.append(reference, sample).astype('int32'))

def target_func(alph, fA, fB, fC):
    return np.linalg.norm(fC - (alph * fA + (1-alph) * fB))

def single_distribution(treeseq, reference, sample):
    subts = get_sub_ARG(treeseq, reference, sample)
    newsample = len(subts.samples()) - 1
    vector = np.zeros(vecsize, dtype=np.float64)
    h_single = np.zeros(vecsize, dtype=np.float64)
    g_single = np.zeros(long_vecsize, dtype=np.float64)
    for tree in subts.trees():
        siblings = tree.siblings(newsample)
        nearest = [[leaf for leaf in tree.leaves(sibling)] for sibling in siblings][0]
        if newsample in nearest:
            nearest.remove(newsample)
        nearest = np.array(nearest)
        refA_neighbours = nearest[nearest < ploidy * k_reference].size
        vector[refA_neighbours] += tree.span * nearest.size
        h_single[refA_neighbours] += 1
        g_single[nearest.size] += 1
    vector /= subts.sequence_length
    h_single /= subts.num_trees
    g_single /= subts.num_trees
    return vector, h_single, g_single

def mean_distribution(treeseq, reference, samples):
    vector = np.zeros(vecsize, dtype=np.float64)
    h_mean = np.zeros(vecsize, dtype=np.float64)
    g_mean = np.zeros(long_vecsize, dtype=np.float64)
    for el in samples:
        sss = single_distribution(treeseq, reference, el)
        vector += sss[0]
        h_mean += sss[1]
        g_mean += sss[2]
    vector /= len(samples)
    h_mean /= len(samples)
    g_mean /= len(samples)
    return vector, h_mean, g_mean

def get_vectors(arg, arg_sets_func):
    reference, learnA, learnB, validation_set = arg_sets_func(arg)
    ret_table = np.zeros((3, vecsize), dtype=np.float64)
    h_table = np.zeros((3, vecsize), dtype=np.float64)
    g_table = np.zeros((3, long_vecsize), dtype=np.float64)
    mA = mean_distribution(arg, reference, learnA)
    mB = mean_distribution(arg, reference, learnB)
    mC = mean_distribution(arg, reference, validation_set)
    ret_table[0] = mA[0]
    ret_table[1] = mB[0]
    ret_table[2] = mC[0]
    h_table[0] = mA[1]
    h_table[1] = mB[1]
    h_table[2] = mC[1]
    g_table[0] = mA[2]
    g_table[1] = mB[2]
    g_table[2] = mC[2]
    return ret_table, h_table, g_table

def get_arg_stats(ARG):
    ans = np.zeros(7, dtype=np.float64)
    span_array = [tree.span for tree in ARG.trees()]
    ans[0] = np.mean(span_array)
    ans[1] = np.var(span_array)
    ans[2] = np.median(span_array)
    ans[3] = ARG.num_trees
    ans[4] = ARG.num_edges
    ans[5] = ARG.num_nodes
    ans[6] = ARG.nbytes
    return ans

def worker(tup):
    debug = False
    alpha, idx, ui = tup
    f_tensor = np.zeros((2, 3, vecsize), dtype=np.float64)
    g_tensor = np.zeros((2, 3, vecsize), dtype=np.float64)
    h_tensor = np.zeros((2, 3, long_vecsize), dtype=np.float64)
    arg_stats_array = np.zeros((2, 7) , dtype=np.float64)
    
    ARG = msprime.sim_ancestry(
        ploidy=ploidy,
        samples=smp,
        demography=dem,
        sequence_length=sequence_len,
        recombination_rate=recomb_rate,
    )
    ARG = msprime.sim_mutations(ARG, rate=mut_rate)
    
    name = f"simulation-v{idx}-u{ui}"
    vcf_name = f"vcf/{name}.vcf"
    with bgzf.open(vcf_name, "w") as f:
        ARG.write_vcf(f,
                     position_transform = lambda x: np.fmax(1, np.round(x))
                     )
    subprocess.run(["tabix", vcf_name])
    ret = subprocess.run(
        [python, "-m", "bio2zarr", "vcf2zarr", "convert", "--force", vcf_name, f"vcf/{name}.vcz"],
        stderr = subprocess.DEVNULL
    )

    schema = json.dumps(tskit.MetadataSchema.permissive_json().schema).encode()
    zarr.save(f"vcf/{name}.vcz/populations_metadata_schema", schema)
    metadata = [
        json.dumps({"name": pop, "description": "The set this individual belongs to"}).encode()
        for pop in types
    ]
    zarr.save(f"vcf/{name}.vcz/populations_metadata", metadata)
    
    num_individuals = ARG.num_individuals
    individuals_population = np.full(num_individuals, tskit.NULL, dtype=np.int32)
    individuals_population[:k_reference] = 0
    individuals_population[k_reference:k_reference+k_learning] = 1
    individuals_population[k_reference+k_learning:2*k_reference+k_learning ] = 2
    individuals_population[2*k_reference+k_learning:2*k_reference+2*k_learning ] = 3
    individuals_population[2*k_reference+2*k_learning: ] = 4
    zarr.save(f"vcf/{name}.vcz/individuals_population", individuals_population)

    ancestral_state = np.array([s.ancestral_state for s in ARG.sites()]) #np.load(f"npy/{name}-AA.npy")
    vdata = tsinfer.VariantData(f"vcf/{name}.vcz",
                                ancestral_state,
                                individuals_population="individuals_population")
    arg_hat = tsinfer.infer(vdata)
    if not debug:
        rmtree(f"vcf/{name}.vcz")
        os.remove(f"vcf/{name}.vcf")
        os.remove(f"vcf/{name}.vcf.tbi")

    vectors = get_vectors(ARG, natural_arg_sets)
    vectors_hat = get_vectors(arg_hat, infered_arg_sets)
    f_tensor[0] = vectors[0]
    f_tensor[1] = vectors_hat[0]
    g_tensor[0] = vectors[1]
    g_tensor[1] = vectors_hat[1]
    h_tensor[0] = vectors[2]
    h_tensor[1] = vectors_hat[2]
    arg_stats_array[0] = get_arg_stats(ARG)
    arg_stats_array[1] = get_arg_stats(arg_hat)
    return f_tensor, g_tensor, h_tensor, arg_stats_array

def universe_worker(uni_id):
    alpha_results = np.zeros((2, 3, vecsize), dtype=np.float64)
    g_results = np.zeros((2, 3, vecsize), dtype=np.float64)
    h_results = np.zeros((2, 3, long_vecsize), dtype=np.float64)
    arg_stats_results = np.zeros((2, 7), dtype=np.float64)
    
    inpt = [(alphas[0], ai, uni) for ai in range(replicates)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        many_tuples = list(executor.map(worker, inpt))
        alpha_results = np.mean(np.array([tup[0] for tup in many_tuples]), axis=0)
        g_results = np.mean(np.array([tup[1] for tup in many_tuples]), axis=0)
        h_results = np.mean(np.array([tup[2] for tup in many_tuples]), axis=0)
        arg_stats_results = np.mean(np.array([tup[3] for tup in many_tuples]), axis=0)

    save_name = f"middle_numpy/u{uni_id}_v{ver}"
    np.save(f"{save_name}_alpha_res.npy", alpha_results)
    np.save(f"{save_name}_g_res.npy", g_results)
    np.save(f"{save_name}_h_res.npy", h_results)
    np.save(f"{save_name}_arg_stats.npy", arg_stats_results)
    estimation_results = np.zeros(2, dtype=np.float64)
    fnA = alpha_results[0][0]
    fnB = alpha_results[0][1]
    fnC = alpha_results[0][2]
    fiA = alpha_results[1][0]
    fiB = alpha_results[1][1]
    fiC = alpha_results[1][2]
    estimation_results[0] = minimize(target_func, x0=start_guess, args=(fnA, fnB, fnC), method='COBYQA', bounds=bnd).x[0]
    estimation_results[1] = minimize(target_func, x0=start_guess, args=(fiA, fiB, fiC), method='COBYQA', bounds=bnd).x[0]
    return estimation_results 

ver=6
# uni = 7
for uni in range(100):
    print(f"universe {uni}")
    es_res = universe_worker(uni_id=uni)
    save_name = f"u{uni}_v{ver}_alphas.npy"
    np.save(save_name, es_res)

