# Design Document

## Overview

The project cleanup system will systematically identify and remove unnecessary files from the debt collection ML project. The cleanup will be organized into categories based on file types and purposes, ensuring that essential files are preserved while removing redundant, temporary, and outdated artifacts.

## Architecture

The cleanup process follows a categorized approach:

1. **File Classification System**: Categorize files based on their purpose and necessity
2. **Safety Validation**: Verify files are safe to delete before removal
3. **Batch Removal**: Remove files in logical groups to maintain project integrity
4. **Verification**: Confirm successful removal and project functionality

## Components and Interfaces

### File Categories

#### Category 1: DVC Fix Scripts
- **Files**: `complete_dvc_fix.bat`, `fix_all_dvc_conflicts.bat`, `nuclear_dvc_fix.bat`, `smart_dvc_fix.bat`, `fix_dvc_and_run.ps1`
- **Rationale**: These are troubleshooting scripts created during DVC setup issues and are no longer needed
- **Safety**: Safe to remove as they don't affect core functionality

#### Category 2: Virtual Environment
- **Files**: `myenv/` directory and all contents
- **Rationale**: Virtual environments should not be committed to version control
- **Safety**: Safe to remove as it can be recreated with `python -m venv myenv`

#### Category 3: Temporary Optimization Results
- **Files**: All timestamped files in `optimization_results/`
  - `*_best_params_*.json`
  - `*_study_*.pkl` 
  - `*_trials_*.csv`
- **Rationale**: These are temporary artifacts from hyperparameter tuning experiments
- **Safety**: Safe to remove as final optimized models are stored in `models/optimized/`

#### Category 4: Redundant Run Scripts
- **Files**: `run_complete_analysis.py`, `run_explainability_analysis.py`, `test_data_generation.py`, `test_environment.py`
- **Rationale**: Functionality has been consolidated into main pipeline scripts
- **Safety**: Safe to remove as functionality exists in `run_enhanced_pipeline.py` and `run_optimized_pipeline.py`

#### Category 5: Temporary Setup Files
- **Files**: `setup_dagshub_repo.py`, `setup_quick.py`, `dagshub_config.py`, `.dagshub_config.json`
- **Rationale**: These were used for initial project setup and are no longer needed
- **Safety**: Safe to remove as DagsHub integration is now handled in `src/utils/dagshub_integration.py`

#### Category 6: Empty/Placeholder Directories
- **Files**: Directories containing only `.gitkeep` files
  - `logs/` (empty)
  - `references/` (only `.gitkeep`)
- **Rationale**: These directories serve no current purpose
- **Safety**: Safe to remove empty directories; preserve structure where needed

#### Category 7: Legacy Configuration Files
- **Files**: `tox.ini`, `pipeline.log`, `dvc_dagshub_pipeline.py`
- **Rationale**: Legacy testing configuration and temporary log files
- **Safety**: Safe to remove as current configuration is in `dvc.yaml` and `params.yaml`

### Preserved Files

#### Essential Core Files
- `run_complete_pipeline.py`, `run_enhanced_pipeline.py`, `run_optimized_pipeline.py`
- `streamlit_dashboard.py`
- `test_explainability.py`
- `dvc.yaml`, `params.yaml`
- `setup.py` (standard Python package setup)

#### Documentation and Configuration
- `README.md`, `DELIVERABLES_SUMMARY.md`
- `docs/` directory
- `.gitignore`, `.env.example`
- `requirements.txt`

#### Source Code and Models
- `src/` directory (all source code)
- `models/` directory (trained models)
- `data/` directory (processed data)

## Data Models

### File Classification Model
```python
class FileCategory:
    name: str
    files: List[str]
    rationale: str
    safety_level: str  # "safe", "caution", "critical"
    dependencies: List[str]
```

### Cleanup Operation Model
```python
class CleanupOperation:
    category: FileCategory
    files_to_remove: List[str]
    backup_required: bool
    verification_steps: List[str]
```

## Error Handling

### File Access Errors
- **Issue**: Files may be locked or in use
- **Solution**: Retry mechanism with user notification
- **Fallback**: Skip locked files and report at end

### Permission Errors
- **Issue**: Insufficient permissions to delete files
- **Solution**: Request elevated permissions or skip with warning
- **Fallback**: Generate list of files that need manual removal

### Dependency Validation
- **Issue**: Accidentally removing files that are still referenced
- **Solution**: Cross-reference with `dvc.yaml`, leanry is cositorm git repon
- Confiperati dashboard o Verifynality
-ine functioipelon
- Test prificati Ve4:hase ## P

#7)(Category n files onfiguratioove legacy c6)
- Remory (Categories ty direct Clean empanup
- CleStructurese 3: Pha
### 
(Category 5)tup files porary seove tem
- Remy 4)ategorts (C scripant runredundRemove 
- dationript Consoli2: Sc## Phase  3)

#(Categoryts tion resulizatimorary optemp Remove )
-(Category 2environment e virtual 
- Remov 1)s (Categoryscriptove DVC fix vals
- Remafe Remose 1: S

### Phahasestion P Implementa##needed

f rtifacts iy ae temporar recreate topipelin-run  Reation**:egener*Rated
3. *cre if ackupe from bre**: RestorestoBackup Rfiles
2. **acked  deleted traccidentally to restore itet**: Use g1. **Git Resategy
k StrRollbaces

### l launchstil dashboard m StreamlitConfirrd Test**: oa. **Dashbe
3olvtill reson imports sl Pytherify alest**: Vmport Trks
2. **I still wore pipelinesuro` to en rep Run `dvc**:eline Test*Pip
1. *fication Verieanup-Cl
### Postlean
s cdirectory irking ure wo*: Enstus Check* **Git Sta3. needed
iles ifal fof criticeate backup Creation**: kup CrBacion
2. **onfiguratd ccode anferences in refile  for *: Scancy Check***Dependenon
1. p ValidatiCleanu Pre-

###gytrate Sestingl

## Tore removackup bef ba**: Create**Fallbackts
- t statemennd imporxt`, aements.tirqu`re