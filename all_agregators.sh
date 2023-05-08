#!/bin/bash
sbatch agregator_job.job ./andmed/const_reverse ./andmed const_reverse.csv
sbatch agregator_job.job ./andmed/sin_reverse ./andmed sin_reverse.csv
sbatch agregator_job.job ./andmed/const_order ./andmed const_order.csv
sbatch agregator_job.job ./andmed/sin_order ./andmed sin_order.csv
