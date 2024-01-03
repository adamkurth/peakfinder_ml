---
runme:
  id: 01HK8HV8JYYZPJ0GX3M15E7PVD
  version: v2.0
---

# peakfinder_ml

Will be ongoing project to implement machine learning techniques for peakfinding in crystallography experimental imaging data.

1. Please ensure that you're logged into AGAVE **first**.

2. Use AGAVE to run `ccn_test.py` using the `run_ccn_SLURM.sh` script.
    
    - Use *gpu* if available for significantly faster computation.

    ``` bash
    ./run_ccn_SLURM.sh <RUN> <TASKS> <PARTITION> <QOS> <HOURS> <TAG>
    ```

3. Then use this command to watch the job:
   
   - Copy the given `JOB_ID` and paste this 
    
    ``` bash
    watch -n 2 squeue -j <JOB_ID>
    ``` 

    - The output files are: 
      - errors: (`.err`)  
      - slurm: (`.slurm`)
      - output: (`.out`)
  
