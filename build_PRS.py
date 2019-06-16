#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
import platform
import subprocess

import numpy as np
import pandas as pd
from pandas import DataFrame
from pyplink import PyPlink

from datetime import datetime


# **defining functions**

def read_discovery_trait_file(path, trait):
    """
    Read the discovery weights trait file and changing the columns names accordingly to the projectmine files
    :param path: string, full path to discovery trait files
    :param trait:  string
    :return:
    """
    print('Start read_discovery_trait_file', datetime.now())

    df = pd.read_csv(path, sep='\t')

    # TODO: take care of fliped effect

    df = handle_trait_file_effect_size(df, trait)

    weights = rename_trait_cols(df, trait)

    # create a new column "ID", which is a identifier for a specific variant
    weights = create_variant_identifier_column(df=weights)

    print('Finish read_discovery_trait_file', datetime.now())

    return weights


def handle_trait_file_effect_size(df, trait):
    """

    :param trait: the name of the trait
    :param df: pandas Data Frame
    :return:
    """

    print('Start handle_trait_file_effect_size', datetime.now())

    binary_traits = ["ADHD", "Alzheimer", "Anorexia", "ASD_PRS", "OCD", "Psoriasis"]
    # if binary trait than working with beta and not OR
    if trait in binary_traits and 'OR' in df.columns:

        df['weight'] = np.log10(df['OR'].values)

        # TODO: specify in the cases, if trait quantities then... if binary then..

    elif 'beta' in  df.columns:
        df['weight'] = df['beta']

    elif 'Beta' in  df.columns:
        df['weight'] = df['Beta']

    elif 'BETA' in  df.columns:
        df['weight'] = df['BETA']

    print('Finish handle_trait_file_effect_size', datetime.now())

    return df


def rename_trait_cols(df, trait=None):
    """
    :param df: pandas.DataFrame
    :param trait: string
    :return: weights, DataFrame
    """

    if trait is not None:
        if trait == 'ADHD':
            # take only relevant columns
            weights = df[['CHR', 'BP', 'weight', 'P', 'A1', 'A2']]
            old_names = ['CHR', 'BP', 'weight', 'P', 'A1', 'A2']
            new_names = ['chr', 'BP', 'weight', 'P', 'A1', 'A2']
            weights.rename(columns=dict(zip(old_names, new_names)), inplace=True)

    return weights


def create_variant_identifier_column(df):
    """
    Create a new column, such as "chr21:9411347:G:C"
    :param df:
    :return:
    """
    print('Start create_variant_identifier_column', datetime.now())

    df.loc[:,'ID'] = 'chr' + df['chr'].map(str) + ':' + df['BP'].map(str) + ":" + df['A2'].map(str) + ":" + df['A1'].map(str)

    print('Finish create_variant_identifier_column', datetime.now())
    return df


def target_PRS_calc(traits_df, file_path):
    """
    TODO: document the func
    1. taking summary statistics of trait gwas
    2. merging with bim data frame to get their intesction with P threshold
    3.
    :param traits_df: pandas.DataFrame, pre-calculated weights from GWAS study
    :return:
    """

    print('Start target_PRS_calc ..', datetime.now())

    n = None
    num_traits = traits_df.shape[1]

    # extract the chromosome to run on based on PRS weights
    trait_chrs = np.unique(traits_df['chr'])

    # build the final PRS vec
    total_PRS_scores = np.zeros((1, 6197), dtype=np.float64)

    # loading PLINK data of a chromosome, one by one
    for CH in trait_chrs:

        # path to ProjectMiNE data
        CH_path = file_path + "nancyy/data/ProjectMiNE/WGS_datafreeze1/WGS_datafreeze1/WGS_datafreeze1_chr" + str(CH) + "_maf_0.0001"

        (bed, bim, fam) = load_plink_data(CH_path)

        if n is None:
            n = fam.shape[0]

        # intersection of variants' weights from the discovery GWAS and the Target ProjectMiNE dataset
        intersec_SNP = merge_discovery_and_target(traits_df, bim)

        # p-value thresholding
        intersec_SNP = pvalue_thresholding(df=intersec_SNP, p=0.1)

        # index number of SNPs
        i = 0

        # chr_PRS is the matrix of the scores of a given chr
        chr_PRS = np.zeros((num_traits, n), dtype=np.float64)

        # get overlapping SNP IDs from the trait and bim
        #snp_names = set(intersec_SNP.index)

        # numpy array
        ## TODO: delete? snp_names = intersec_SNP.index.values

        print("Strating to iterate on bed file.. ", datetime.now())

        # iterate on the bed file by columns (in a lazy manner)
        for loci_name, geno in bed:

            # change the name from the form of 'chr21:14840569:G:A' to 14840569
            # loci_name = change_snp_column_values(loci_name)

            #print("if loci_name in snp_names:", datetime.now())
            # checking if the variant is a valid one

            if short_surch(loci_name, intersec_SNP) and long_surch(loci_name, intersec_SNP):
                #print("if loci_name in snp_names:", i, datetime.now())

                # get the weight
                variant_weight = intersec_SNP.loc[loci_name, 'weight']

                # get the value of the genotype -1,1,2
                # extract only the positions of non-zero
                # the bed matrix is very sparse. This step is necessary,
                # in order to handle the big dimension of the data
                geno_nonzero_ind = np.nonzero(geno)[0]
                g = geno[geno_nonzero_ind]

                # making the -1 to be as 0 for calculation
                g = np.where(g < 0, 0, g)

                # additing the samples scores
                # #TODO: this is for current trait, but need to change chr_PRS[0,X] 0 should be index
                chr_PRS[0, geno_nonzero_ind] += (g * variant_weight)

                # ----------------------#
                # showing development of the loop
                i += 1
                if i % 10000 == 0:
                    print("Finished 10^4 SNPs", datetime.now())

        total_PRS_scores += chr_PRS[0]

        print(" *************** Finished chromosome" + str(CH) + "!! *************** ")

    total_PRS_scores = pd.DataFrame(data=total_PRS_scores, columns=fam['iid'], index=['PRS_score'])

    print('Finish target_PRS_calc ..', datetime.now())

    return total_PRS_scores


def load_plink_data(path):
    """
    Load the WGS data BED files for a giving chromosome
    Read PLINK files (bim, fam, bed) into pandas data frames using the PyPlink package (allow to read single file at a time)

    Note:  PyPlink does not load the BED file in memory, making possible to work with extremely large files and iterate on them

    n = number of subjects
    p = number of variants

    :param path: string, path to binary PLINK files

    :return bed: numpy matrix (shape: n x p)
            bim: pandas.DataFrame (p x 5)
            fam: pandas.DataFrame (n X 6)
   """
    print('Start loading the PLINK files', datetime.now())

    bed = PyPlink(path)

    # --------------------------------
    #  Getting the BIM and FAM
    fam = bed.get_fam()  # Samples. Returns the FAM file as pandas.DataFrame
    bim = bed.get_bim()  # Alleles. Returns the BIM file as pandas.DataFrame with changed colums order and names

    bim = handle_bim_columns(bim)

    return bed, bim, fam


def handle_bim_columns(bim):
    """

    :param bim:
    :return:
    """

    cols = bim.columns.tolist()
    cols = cols[0:1] + cols[3:4] + cols[2:3] + cols[1:2] + cols[-2:]
    bim = bim.reindex(cols, axis=1)
    old_names = cols
    new_names = ['chr', 'SNP', 'CM', 'BP', 'A1', 'A2']
    bim.rename(columns=dict(zip(old_names, new_names)), inplace=True)
    return bim


def merge_discovery_and_target(weights_df, bim_df):
    """
    Get the trait weights and bim and find the intersection
    Returning DataFrame with variant ID and their weights
    Note: the index is according to the **bim file**
    :param weights_df: DataFrame
    :param bim_df: DataFrame
    :return: df DateFrame
    """

    print('Start merge_discovery_and_target', datetime.now())

    print("# variants in discovery trait data: ", weights_df.shape)
    print("# variants in target ProjectMiNE Bim : ", bim_df.shape)

    # merge on 'snp' (the index of bim) and the column 'ID' in trait file

    df = pd.merge(bim_df, weights_df, how='inner', left_index=True, right_on=['ID'], copy=False).set_index('ID')

    # verify there are no duplicates
    df = df[~df.index.duplicated(keep='first')]
    # TODO: print(df.columns) and keep only relevant columns df = df[[]]
    print("After merge and droping duplication, remaining: ", df.shape)

    print('Finish merge_discovery_and_target', datetime.now())

    return df


def pvalue_thresholding(df, p=0.05):

    print('Start pvalue_thresholding', datetime.now())

    print("Before p-value thresholding: ", df.shape)

    df = df.loc[df['P'] >= p, :]

    print("After p-value thresholding we remain with: ", df.shape)

    print('Finish p-value_thresholding', datetime.now())

    return df


def change_snp_column_values(my_string):
    """
    Change SNP column values to hold only location. Used when iterating on bed matrix
    E.g., change the value from
            my_string='chr21:14840569:G:A'
            to
            snp_loc='14840569'
    :param my_string: string
    :return: snp_loc: int
    """
    # Find the first ":", and remove what ever is before it
    my_string = my_string[my_string.find(":")+1:]
    # Find the second ":", and remove what ever is after it
    snp_loc = int(my_string[:my_string.find(":")])
    return snp_loc


def short_surch(loci, snp_bp):
    """
    fast surch - alawing shor sircuit
    :param loci: string, the full snp name
    :param snp_bp: dataframe vector of the BP
    :return: bulian T or F
    """
    loci_bp = change_snp_column_values(loci)
    #print("this is in the short - num: ", loci_bp, "in", snp_bp[0:5])
    is_it_there = loci_bp in set(snp_bp['BP_x'].values)

    return is_it_there

def long_surch(loci, snp_id):
    """
    slow didactic surch
    :param loci: string, the full snp name
    :param snp_id: dataframe vector of the ID
    :return: bulian T or F
    """
    is_it_there = loci in set(snp_id.index)

    return is_it_there

# **EXECUTION**

def main(main_path=None, PRSs_path=None, trait=None):
    """
    1. Load pre-calculated weights from a GWAS study discovery trait file
    2. Calculate PRSs scores for individuals in ProjectMine dataset
    3. Save PRSs scores to file
    :param trait: string
    :return:
    """

    # TODO: add loop that will run on th traits files

    print('Start calculating PRSs..', datetime.now())

    if trait == 'ADHD':


        trait_path = PRSs_path + "ADHD/adhd_jul2017"

        weights_df = read_discovery_trait_file(trait_path, trait)

        # calculating PRS
        PRS = target_PRS_calc(weights_df, main_path)

        # saving as vec
        file_path = main_path + "noamha/ProjectMiNE/" + trait + "_PRS_.csv"
        PRS.to_csv(file_path, index=True)

    else:
        print("Trait name was not provided. Finish running..")

    print('Finish calculating PRSs..', datetime.now())


if __name__ == "__main__":

    os = platform.system()
    # MOLGEN OR LOCAL RUN - CHANGE HERE
    if os == 'Windows':
        main_path = r"Z:/"
    else:
        main_path = r"/home/labs/hornsteinlab/"
    PRSs_path = main_path + "nancyy/data/PRSsumstat/"

    main(main_path, PRSs_path, trait='ADHD')