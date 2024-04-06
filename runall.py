#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from utils import bulk_create_and_store_bundles, bulk_cost_estimate, new_bulk_cost_estimate
from cost import all_pairs, random_buckets, list_decoding, sieve_size, Metrics, SizeMetrics
from texconstsf import main as texconstsff

import argparse

# Add required command line arguments
CLI=argparse.ArgumentParser()
CLI.add_argument(
  "--ds", 
  nargs="*",  
  type=int,
  default=[375, 586, 829, 394, 587, 818], 
  help="List of dimensions",
)
CLI.add_argument(
  "--depths",
  nargs="*",
  type=int,  
  default=[1,2,3],
  help="List of recursion depths",
)
CLI.add_argument(
    "--metric",
    nargs="*",
    type=str,
    default="hardware_time",
    help="List or single metric from 'hardware_time' or 'classical'",
)
CLI.add_argument(
    "--mem_cost",
    nargs="*",
    type=float,
    default = 2.0**(-12.8),
    help = "Coefficient for cost of sort",
)
CLI.add_argument(
    "--jobs",
    type=int,
    default=1,
    help = "Number of threads to use",
)
CLI.add_argument(
    "--approximate",
    default=True,
    help = "Compute approximate costs only",
    action='store_false',
)

CLI.add_argument(
    "--mem_exp",
    nargs="*",
    type=float,
    default=0.5,
    help="Memory parameter Delta",
)

CLI.add_argument(
    "--exhaustive",
    type=int,
    default=0,
    help="Number of intervals for exhaustive theta search (0 does binary search)"
)

# Parse and apply
args = CLI.parse_args()
new_bulk_cost_estimate(
    mem_cost = args.mem_cost, 
    mem_exponent = args.mem_exp,
    depths = args.depths, 
    D = args.ds, 
    metric = args.metric, 
    filename = None, 
    exact = args.approximate, 
    ncores = args.jobs,
    exhaustive_size=args.exhaustive)

