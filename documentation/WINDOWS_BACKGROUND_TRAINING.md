# Windows Background Training Guide

This guide shows how to run nnU-Net training in background PowerShell sessions on Windows (equivalent to Linux `screen` command).

## Quick Reference

### Basic Command Pattern

```powershell
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\path\to\three_types'; .venv\Scripts\python.exe scripts/segmentation/nnunet_training.py --dataset_id 100 --dataset_name Liver1 --config train_config.json --trainer nnUNetTrainer_500epochs --fold_range 0-9 --random_seed 42" -WindowStyle Minimized
```

### With Logging (Recommended)

```powershell
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\path\to\three_types'; .venv\Scripts\python.exe scripts/segmentation/nnunet_training.py --dataset_id 100 --dataset_name Liver1 --config train_config.json --trainer nnUNetTrainer_500epochs --fold_range 0-9 --random_seed 42 2>&1 | Tee-Object training_100.log" -WindowStyle Minimized
```

## Training All 5 Partitions

### Start All Training Sessions

```powershell
# Partition 1 (Dataset100_Liver1)
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\path\to\three_types'; .venv\Scripts\python.exe scripts/segmentation/nnunet_training.py --dataset_id 100 --dataset_name Liver1 --config train_config.json --trainer nnUNetTrainer_500epochs --fold_range 0-9 --random_seed 42 2>&1 | Tee-Object training_100.log" -WindowStyle Minimized

# Partition 2 (Dataset101_Liver2)
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\path\to\three_types'; .venv\Scripts\python.exe scripts/segmentation/nnunet_training.py --dataset_id 101 --dataset_name Liver2 --config train_config.json --trainer nnUNetTrainer_500epochs --fold_range 0-9 --random_seed 42 2>&1 | Tee-Object training_101.log" -WindowStyle Minimized

# Partition 3 (Dataset102_Liver3)
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\path\to\three_types'; .venv\Scripts\python.exe scripts/segmentation/nnunet_training.py --dataset_id 102 --dataset_name Liver3 --config train_config.json --trainer nnUNetTrainer_500epochs --fold_range 0-9 --random_seed 42 2>&1 | Tee-Object training_102.log" -WindowStyle Minimized

# Partition 4 (Dataset103_Liver4)
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\path\to\three_types'; .venv\Scripts\python.exe scripts/segmentation/nnunet_training.py --dataset_id 103 --dataset_name Liver4 --config train_config.json --trainer nnUNetTrainer_500epochs --fold_range 0-9 --random_seed 42 2>&1 | Tee-Object training_103.log" -WindowStyle Minimized

# Partition 5 (Dataset104_Liver5)
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\path\to\three_types'; .venv\Scripts\python.exe scripts/segmentation/nnunet_training.py --dataset_id 104 --dataset_name Liver5 --config train_config.json --trainer nnUNetTrainer_500epochs --fold_range 0-9 --random_seed 42 2>&1 | Tee-Object training_104.log" -WindowStyle Minimized
```

## Monitoring Training

### View Real-Time Log Output

```powershell
# Watch Partition 1
Get-Content training_100.log -Wait -Tail 50

# Watch Partition 2
Get-Content training_101.log -Wait -Tail 50

# Watch Partition 3
Get-Content training_102.log -Wait -Tail 50

# Watch Partition 4
Get-Content training_103.log -Wait -Tail 50

# Watch Partition 5
Get-Content training_104.log -Wait -Tail 50
```

Press `Ctrl+C` to stop watching (doesn't stop training, just stops viewing).

### List All Running PowerShell Processes

```powershell
Get-Process powershell
```

### Find Specific Training Process

```powershell
Get-Process powershell | Where-Object {$_.MainWindowTitle -like "*nnunet*"}
```

## Window Styles

You can change `-WindowStyle` to:
- `Normal` - Regular window
- `Minimized` - Minimized to taskbar (recommended)
- `Hidden` - Completely hidden (no taskbar icon)

## Stopping Training

### Find Process ID

```powershell
Get-Process powershell
```

Look for the process with high CPU usage or check window title.

### Stop Specific Process

```powershell
Stop-Process -Id <PID>
```

### Stop All Training Processes (CAREFUL!)

```powershell
Get-Process powershell | Where-Object {$_.MainWindowTitle -like "*nnunet*"} | Stop-Process
```

## Training Specific Folds

### Single Fold

```powershell
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\path\to\three_types'; .venv\Scripts\python.exe scripts/segmentation/nnunet_training.py --dataset_id 100 --dataset_name Liver1 --config train_config.json --trainer nnUNetTrainer_500epochs --fold 0 --random_seed 42 2>&1 | Tee-Object training_100_fold0.log" -WindowStyle Minimized
```

### Fold Range

```powershell
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\path\to\three_types'; .venv\Scripts\python.exe scripts/segmentation/nnunet_training.py --dataset_id 100 --dataset_name Liver1 --config train_config.json --trainer nnUNetTrainer_500epochs --fold_range 0-4 --random_seed 42 2>&1 | Tee-Object training_100_folds0-4.log" -WindowStyle Minimized
```

## Alternative: PowerShell Jobs

If you prefer jobs over separate windows:

### Start Job

```powershell
Start-Job -Name "partition1" -ScriptBlock {
    cd "C:\path\to\three_types"
    & ".venv\Scripts\python.exe" nnunet_training.py --dataset_id 100 --dataset_name Liver1 --config train_config.json --trainer nnUNetTrainer_500epochs --fold_range 0-9 --random_seed 42 2>&1 | Tee-Object -FilePath "training_100.log"
}
```

### Monitor Job

```powershell
# Check status
Get-Job

# View output (keeps job running)
Receive-Job -Name "partition1" -Keep

# Get final output and remove job
Receive-Job -Name "partition1"
Remove-Job -Name "partition1"
```

### Stop Job

```powershell
Stop-Job -Name "partition1"
Remove-Job -Name "partition1"
```

## Tips

1. **Use logging (`Tee-Object`)** - Makes it easy to check progress later
2. **Use `-WindowStyle Minimized`** - Keeps windows out of the way
3. **Name your log files clearly** - `training_<dataset_id>.log` or `training_<dataset_id>_fold<N>.log`
4. **Monitor GPU usage** with Task Manager or `nvidia-smi`
5. **Check disk space** - nnU-Net creates large preprocessed datasets

## Troubleshooting

### Process Won't Start

Check if Python path is correct:
```powershell
Test-Path ".venv\Scripts\python.exe"
```

### Can't Find Log File

Logs are created in the directory where you run the command. Check:
```powershell
Get-ChildItem *.log
```

### Training Crashes

Check the log file for errors:
```powershell
Get-Content training_100.log -Tail 100
```

### Out of Memory

Reduce batch size in `train_config.json` or train folds sequentially instead of all at once.

## Example: Sequential Fold Training

If you want to train folds one at a time (to manage memory):

```powershell
# Train fold 0
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\path\to\three_types'; .venv\Scripts\python.exe scripts/segmentation/nnunet_training.py --dataset_id 100 --dataset_name Liver1 --config train_config.json --trainer nnUNetTrainer_500epochs --fold 0 --random_seed 42 2>&1 | Tee-Object training_100_fold0.log" -WindowStyle Minimized -Wait

# After fold 0 completes, train fold 1
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\path\to\three_types'; .venv\Scripts\python.exe scripts/segmentation/nnunet_training.py --dataset_id 100 --dataset_name Liver1 --config train_config.json --trainer nnUNetTrainer_500epochs --fold 1 --random_seed 42 2>&1 | Tee-Object training_100_fold1.log" -WindowStyle Minimized -Wait

# And so on...
```

The `-Wait` flag makes PowerShell wait for the process to complete before continuing.
