@echo off
echo Fixing ALL DVC conflicts and running dvc repro...

echo Step 1: Removing ALL conflicting files and directories from Git tracking...
git rm -r --cached "models\artifacts" 2>nul
git rm -r --cached "models\trained" 2>nul
git rm -r --cached "data\processed" 2>nul
git rm -r --cached "data\features" 2>nul
git rm -r --cached "explanations" 2>nul
git rm -r --cached "reports\metrics.json" 2>nul
git rm -r --cached "reports\enhanced_pipeline_report.txt" 2>nul
git rm -r --cached "reports\model_comparison.csv" 2>nul

echo Step 2: Committing Git changes...
git add .
git commit -m "stop tracking ALL DVC-managed files and directories"

echo Step 3: Already updated .gitignore (DVC entries already added)

echo Step 4: Running DVC repro...
dvc repro

echo DVC pipeline completed!
pause