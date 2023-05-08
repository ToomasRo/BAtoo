#!/bin/bash
sbatch const_reverse_runner.job 4
sbatch sin_reverse_runner.job 4
sbatch const_reverse_runner.job 0.25
sbatch sin_reverse_runner.job 0.25
sbatch const_reverse_runner.job 0.5
sbatch sin_reverse_runner.job 0.5
sbatch const_reverse_runner.job 1
sbatch sin_reverse_runner.job 1
sbatch const_reverse_runner.job 2
sbatch sin_reverse_runner.job 2

sbatch const_order_runner.job 4
sbatch sin_order_runner.job 4
sbatch const_order_runner.job 0.25
sbatch sin_order_runner.job 0.25
sbatch const_order_runner.job 0.5
sbatch sin_order_runner.job 0.5
sbatch const_order_runner.job 1
sbatch sin_order_runner.job 1
sbatch const_order_runner.job 2
sbatch sin_order_runner.job 2
