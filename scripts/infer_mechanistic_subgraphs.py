import hydra
from omegaconf import DictConfig
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import accumulate
from collections import defaultdict
from ergochemics.mapping import rc_to_nest, rc_to_str
import logging
from time import perf_counter
from tqdm import tqdm
from cgr.inference import (
    ReactantGraph,
    MolFeaturizer,
    atom_featurizer_v2,
    get_stereotypical_molecules
) 

def bayesian_freq(ct: int, N_i: int, N_avg: int) -> float:
    pass

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path='../configs', config_name='infer_mech_subgraphs')
def main(cfg: DictConfig):    
    rxn_subset = pd.read_parquet(Path(cfg.input_path))
    rxn_subset = rxn_subset.loc[rxn_subset['rule_id'] == cfg.rule_id].reset_index(drop=True)
    mol_featurizer = MolFeaturizer(atom_featurizer_v2)

    lhs_exclude, _ = get_stereotypical_molecules(rxn_subset["smarts"].tolist())

    log.info(f"Processing {len(rxn_subset)} reactions mapped to rule {cfg.rule_id}")

    # Decompose reactions into subgraphs
    tic = perf_counter()
    log.info("Decomposing reactions into subgraphs...")
    n_subgraphs = [{} for _ in range(cfg.max_n)] # Subgraphs of size n at index n - 1
    subgraph_decomp = []
    sg_idx_lut = defaultdict(lambda : [{} for _ in range(cfg.max_n)])  # {rxn_id: {sg_id: sidxs}}
    full_rgs = {}
    for _, elt in rxn_subset.iterrows():
        rxn_id = elt["rxn_id"]
        decomp_rxn = [{} for _ in range(cfg.max_n)]
        rcts = elt["smarts"].split(">>")[0]
        rc = rc_to_nest(elt["reaction_center"])[0]
        rg = ReactantGraph.from_smiles(rcts=rcts, featurizer=mol_featurizer, rc=rc, exclude=lhs_exclude)
        full_rgs[rxn_id] = rg
        subgraph_idxs = rg.k_hop_subgraphs(cfg.k)

        for sidxs in subgraph_idxs:
            subgraph = rg.subgraph(list(sidxs))
            n = len(sidxs) - 1
            nn = len(n_subgraphs[n])
            if nn == 0:
                n_subgraphs[n][0] = subgraph
                decomp_rxn[n][0] = subgraph
                sg_idx_lut[rxn_id][n][0] = sidxs
            else: # Check if subgraph is new to the rxn_subset
                exists = False
                for j, sj in n_subgraphs[n].items():
                    if subgraph == sj:
                        decomp_rxn[n][j] = subgraph
                        sg_idx_lut[rxn_id][n][j] = sidxs
                        exists = True
                        break

                if not exists:
                    n_subgraphs[n][nn] = subgraph
                    decomp_rxn[n][nn] = subgraph
                    sg_idx_lut[rxn_id][n][nn] = sidxs

        subgraph_decomp.append(decomp_rxn)

    toc = perf_counter()
    log.info(f"Decomposing reactions took {toc - tic:.2f} seconds")

    tic = perf_counter()
    log.info("Constructing binary feature matrix and subgraph instances...")
    # Construct & binary feature matrix & subgraph instances
    sgid_offsets = [0] + list(accumulate([len(elt) for elt in n_subgraphs]))
    bfm = np.zeros(shape=(len(subgraph_decomp), sgid_offsets[-1])) # Binary feature matrix
    tmp = []
    for i, decomp_rxn in enumerate(subgraph_decomp):
        for so, n_fts in zip(sgid_offsets, decomp_rxn):
            for k, v in n_fts.items():
                bfm[i, k + so] = 1
                sep_sg_idxs = [v.sep_aidxs[v.rct_idxs == n].tolist() for n in range(v.n_rcts)]

                tmp.append(
                    [
                        k + so,
                        rxn_subset.loc[i, "rxn_id"],
                        rxn_subset.loc[i, "smarts"],
                        rxn_subset.loc[i, "am_smarts"],
                        rxn_subset.loc[i, "reaction_center"],
                        v.aidxs.tolist(),
                        rc_to_str([sep_sg_idxs, [[]]]),
                    ]
                )
    
    # Outputs subgraph indices both for the multi-reactant graph (sg_idxs) and for separate single reactant graphs (sep_sg_idxs)
    sg_insts = pd.DataFrame(tmp, columns=["subgraph_id", "rxn_id", "smarts", "am_smarts", "reaction_center", "sg_idxs", "sep_sg_idxs"])

    toc = perf_counter()
    log.info(f"Constructing binary feature matrix and subgraph instances took {toc - tic:.2f} seconds")
    
    # Save
    print("Saving data...")
    sg_insts.to_parquet("subgraph_instances.parquet")
    np.save("bfm.npy", bfm)

    # Vary frequency lower bound and crossref mechanistic subgraphs
    log.info("Cross-referencing known reactions & mechanistic subgraphs...")
    if cfg.xref:
        # TODO: this is another messy thing resulting from keeping n_node subgraphs separate... maybe not worth it?
        tmp = defaultdict(lambda: defaultdict(tuple))
        for rxn_id, n_sgs in sg_idx_lut.items():
            for n, sgs in enumerate(n_sgs):
                for k, v in sgs.items():
                    tmp[rxn_id][k + sgid_offsets[n]] = v

        sg_idx_lut = tmp

        # Load mapped and mech labeled reactions
        mm = pd.read_parquet(
        Path(cfg.filepaths.raw_data) / "mapped_mech_labeled_reactions.parquet"
        )
        mm = mm.loc[mm['rule_id'] == cfg.rule_id].reset_index(drop=True)
        mm["reaction_center"] = mm["reaction_center"].apply(rc_to_nest)
        mm["mech_atoms"] = mm["mech_atoms"].apply(rc_to_nest)

        mm_lhs_exclude, _ = get_stereotypical_molecules(mm["smarts"].tolist())
        
        # TODO: Messy. Push into new RectantGraph class method or utility function
        # Construct ReacantGraphs for mechanistic subgraphs
        mech_rgs = []
        for _, elt in mm.iterrows():
            rcts = elt["smarts"].split(">>")[0]
            rc = elt["reaction_center"][0]
            rg = ReactantGraph.from_smiles(rcts=rcts, featurizer=mol_featurizer, rc=rc, exclude=mm_lhs_exclude)
            mech_atoms = elt["mech_atoms"][0]
            nidxs = []
            for i, ma in enumerate(mech_atoms):
                if i in mm_lhs_exclude:
                    continue

                for saidx in ma:
                    nidx = np.argwhere(
                        (rg.sep_aidxs == saidx) & (rg.rct_idxs == i)
                    )
                    nidx = int(nidx.flatten()[0])
                    nidxs.append(nidx)

            mech_rg = rg.subgraph(nidxs)
            mech_rgs.append(mech_rg)

        p1 = bfm.sum(axis=0) / bfm.shape[0]
        n_rxns = bfm.shape[0]
        rcsz = full_rgs[list(full_rgs.keys())[0]].rcsz

        mech_cov_cols = ["scl_lb", "mech_id", "inf_id", "coverage", "atom_ratio"]
        summary_stats_cols = ["scl_lb", "n_novel_subgraphs", "n_total_inferred", "rxn_cov_frac"]
        mech_cov_data = []
        summary_stats_data = []

        tic = perf_counter()
        for scl_lb in tqdm(np.arange(1, n_rxns + 1, cfg.ds)):
            rxn_ct = 0
            # Extract inferred subgraphs = union of subgraphs with frequency >= lb
            # on per-reaction basis
            inferred_subgraphs = {}
            lb = scl_lb / n_rxns
            for rxn_id, gb in sg_insts.groupby(by="rxn_id"):

                # Take union of subgraphs with frequency >= lb
                nidxs = set()
                for _, row in gb.iterrows():
                    sg_id = row['subgraph_id']
                    if p1[sg_id] >= lb:
                        nidxs.update(sg_idx_lut[rxn_id][sg_id])

                rg = full_rgs[rxn_id]
                nidx_complement = [i for i in range(rg.V.shape[0]) if i not in nidxs]
                if (rg.V[nidx_complement, -rg.rcsz:] == 0).any(): # Reject if any RC atoms are not covered
                    continue

                rc_mask = (rg.V[:, -rg.rcsz:] == 0).any(axis=1)
                if (rg.V[rc_mask, -(rg.rcsz + 1)] == 0).all(): # Reject if only adds C w/ amphoteros ox state = 0
                    continue

                subgraph = full_rgs[rxn_id].subgraph(list(nidxs))

                rxn_ct += 1 # Count rxn covered
                subgraph.remove_specific_indexing()

                if len(inferred_subgraphs) == 0: 
                    inferred_subgraphs[0] = subgraph
                else:
                    exists = False
                    for j, sj in inferred_subgraphs.items():
                        if subgraph == sj:
                            exists = True
                            break

                    if not exists:
                        inferred_subgraphs[len(inferred_subgraphs)] = subgraph

            # Do MCS on all pairs of (mechanistic, inferred) subgraphs
            # and calculate summary stats
            mcs = defaultdict(lambda: defaultdict(list))
            novels = set(inferred_subgraphs.keys())
            for i, inf_rg in inferred_subgraphs.items():
                inf_n_atoms = inf_rg.V.shape[0]
                for j, mech_rg in enumerate(mech_rgs):
                    mech_n_atoms = mech_rg.V.shape[0]
                    M = list(inf_rg.mcs(mech_rg)) # MCS mapping
                    mcs[i][j] = M

                    # If entire mech is a subgraph of inferred, inferred is not novel
                    # else, is putatively novel
                    if len(M) == mech_n_atoms:
                        novels.discard(i) 

                    # coverage is the ratio of MCS size to mechanistic subgraph size
                    # atom_ratio is the ratio of inferred subgraph size to mechanistic subgraph size
                    coverage = len(M) / mech_n_atoms
                    atom_ratio = inf_n_atoms / mech_n_atoms

                    mech_cov_data.append(
                        [
                            scl_lb,
                            j,
                            i,
                            coverage,
                            atom_ratio
                        ]
                    )

            rxn_cov_frac = rxn_ct / n_rxns
            summary_stats_data.append([scl_lb, len(novels), len(inferred_subgraphs), rxn_cov_frac])

            if rxn_cov_frac == 0:
                toc = perf_counter()
                log.info(f"Halted at scl_lb {scl_lb} with rxn_cov_frac = 0")
                log.info(f"Time taken: {toc - tic:.2f} seconds")
                log.info(f"{scl_lb / (toc - tic):.2f} it/s")
                break

        # Save summary stats
        coverage_df = pd.DataFrame(mech_cov_data, columns=mech_cov_cols)
        summary_stats_df = pd.DataFrame(summary_stats_data, columns=summary_stats_cols)
        basic_df = pd.DataFrame(data=[[n_rxns, rcsz]], columns=["n_rxns", "rcsz"])
        basic_df.to_csv("basic.csv", index=False)
        coverage_df.to_csv("mech_coverage.csv", index=False)
        summary_stats_df.to_csv("summary_stats.csv", index=False)

if __name__ == '__main__':
    main()