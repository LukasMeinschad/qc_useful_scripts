# Useful Slurm and Job Commands

Run all **sbatch** files in a given directory
```{bash}
find . -type f -name "*.sjob" -execdir sbatch {} \;
``` 