import os
from pathlib import Path
import scipy.io
import argparse
import re
from collections import defaultdict

def load_spm_regressor_names(spm_path):
    """Load regressor names from SPM.mat file."""
    try:
        spm_data = scipy.io.loadmat(spm_path, struct_as_record=False, squeeze_me=True)
        spm = spm_data['SPM']
        
        # Extract regressor names (remove 'Sn(X) ' prefix and '*bf(1)' suffix)
        regr_names = []
        for name in spm.xX.name:
            # Remove 'Sn(X) ' prefix
            clean_name = re.sub(r'^Sn\(\d+\) ', '', name)
            # Remove '*bf(1)' suffix if present
            if '*bf(1)' in clean_name:
                clean_name = clean_name.replace('*bf(1)', '')
            regr_names.append(clean_name)
        
        return regr_names
    except Exception as e:
        print(f"Error loading {spm_path}: {e}")
        return None
    
def explore_subject_data(subject_path):
    """Explore data available for a single subject."""
    subject_name = subject_path.name
    info = {
        'subject': subject_name,
        'tasks': [],
        'models': {}
    }
    
    # Look for task directories
    for task_dir in subject_path.iterdir():
        if task_dir.is_dir():
            info['tasks'].append(task_dir.name)
            
            # Look for model directories within each task
            task_models = []
            for item in task_dir.iterdir():
                if item.is_dir() and item.name.startswith('model_'):
                    model_num = item.name.replace('model_', '')
                    task_models.append(model_num)
                    
                    # Check if SPM.mat exists and get regressor info
                    spm_path = item / 'SPM.mat'
                    if spm_path.exists():
                        regressors = load_spm_regressor_names(spm_path)
                        if regressors:
                            info['models'][f"{task_dir.name}_model_{model_num}"] = {
                                'regressors': regressors,
                                'n_regressors': len(regressors),
                                'beta_files': len(list(item.glob('beta_*.nii')))
                            }
            
            if task_models:
                info['models'][task_dir.name] = sorted(task_models)
    
    return info

def print_subject_summary(subject_info):
    """Print a formatted summary for a single subject."""
    print(f"\n{subject_info['subject']}:")
    print(f"  Tasks: {', '.join(subject_info['tasks'])}")
    
    for key, value in subject_info['models'].items():
        if isinstance(value, list):
            print(f"  {key} models: {', '.join(value)}")
        elif isinstance(value, dict):
            print(f"  {key}:")
            print(f"    Regressors: {value['n_regressors']}")
            print(f"    Beta files: {value['beta_files']}")
            
            # Show condition regressors (exclude nuisance)
            condition_regressors = [r for r in value['regressors'] 
                                  if not any(nuis in r.lower() for nuis in 
                                           ['buttonpress', 'constant', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])]
            if condition_regressors:
                print(f"    Condition regressors: {', '.join(condition_regressors[:5])}")
                if len(condition_regressors) > 5:
                    print(f"      ... and {len(condition_regressors) - 5} more")
                    

def check_model_availability(base_dir):
    """Check which subjects have model 28 data available."""
    betas_dir = Path(base_dir) / 'derivatives' / 'spm-preproc' / 'derivatives' / 'spm-stats' / 'betas'
    
    if not betas_dir.exists():
        print(f"Betas directory does not exist: {betas_dir}")
        return []
    
    model_28_subjects = []
    model_28_details = []
    
    for subject_dir in betas_dir.iterdir():
        if subject_dir.is_dir() and subject_dir.name.startswith('sub-'):
            test_model_28 = subject_dir / 'test' / 'model_28'
            if test_model_28.exists():
                model_28_subjects.append(subject_dir.name)
                
                # Get detailed info
                spm_path = test_model_28 / 'SPM.mat'
                if spm_path.exists():
                    regressors = load_spm_regressor_names(spm_path)
                    beta_files = len(list(test_model_28.glob('beta_*.nii')))
                    
                    # Count condition regressors (relevant for MVPA)
                    condition_regressors = []
                    if regressors:
                        condition_regressors = [r for r in regressors 
                                              if not any(nuis in r.lower() for nuis in 
                                                       ['buttonpress', 'constant', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])]
                    
                    model_28_details.append({
                        'subject': subject_dir.name,
                        'total_regressors': len(regressors) if regressors else 0,
                        'condition_regressors': len(condition_regressors),
                        'beta_files': beta_files,
                        'conditions': condition_regressors if condition_regressors else []
                    })
    
    return sorted(model_28_subjects), model_28_details

def main():
    parser = argparse.ArgumentParser(description='Explore fMRI data structure')
    parser.add_argument('--base-dir', type=str, default='/project/3018040.05/bids',
                        help='Base directory to explore (default: /project/3018040.05/bids)')
    parser.add_argument('--subject', '-s', type=str, help='Explore specific subject only')
    parser.add_argument('--check-model-28', action='store_true', 
                        help='Check availability of model 28 data specifically')
    parser.add_argument('--verbose', '-v', action='store_true', 
                        help='Show detailed regressor information')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    betas_dir = base_dir / 'derivatives' / 'spm-preproc' / 'derivatives' / 'spm-stats' / 'betas'
    
    if not betas_dir.exists():
        print(f"Betas directory does not exist: {betas_dir}")
        return
    
    print(f"Exploring data in: {betas_dir}")
    
    if args.check_model:
        print(f"\n{'='*60}")
        print("MODEL 28 AVAILABILITY CHECK")
        print(f"{'='*60}")
        
        model_28_subjects, details = check_model_availability(base_dir)
        
        print(f"Subjects with model 28 data: {len(model_28_subjects)}")
        if model_28_subjects:
            print(f"Subject list: {', '.join(model_28_subjects)}")
            
            if args.verbose and details:
                print(f"\nDetailed breakdown:")
                for detail in details:
                    print(f"\n{detail['subject']}:")
                    print(f"  Total regressors: {detail['total_regressors']}")
                    print(f"  Condition regressors: {detail['condition_regressors']}")
                    print(f"  Beta files: {detail['beta_files']}")
                    if detail['conditions']:
                        print(f"  Conditions: {', '.join(detail['conditions'][:8])}")
                        if len(detail['conditions']) > 8:
                            print(f"    ... and {len(detail['conditions']) - 8} more")
        else:
            print("No subjects found with model 28 data")
    
    else:
        # General exploration
        if args.subject:
            subject_dir = betas_dir / args.subject
            if subject_dir.exists():
                info = explore_subject_data(subject_dir)
                print_subject_summary(info)
            else:
                print(f"Subject {args.subject} not found in {betas_dir}")
        else:
            # Explore all subjects
            subjects = [d for d in betas_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('sub-')]
            
            if not subjects:
                print("No subjects found")
                return
            
            print(f"Found {len(subjects)} subjects")
            
            # Summary statistics
            task_counts = defaultdict(int)
            model_counts = defaultdict(int)
            
            for subject_dir in sorted(subjects):
                info = explore_subject_data(subject_dir)
                
                if args.verbose:
                    print_subject_summary(info)
                
                # Collect statistics
                for task in info['tasks']:
                    task_counts[task] += 1
                
                for key in info['models']:
                    if 'model_' in key:
                        model_counts[key] += 1
            
            print(f"\n{'='*60}")
            print("SUMMARY STATISTICS")
            print(f"{'='*60}")
            print(f"Total subjects: {len(subjects)}")
            print(f"Tasks found: {dict(task_counts)}")
            print(f"Models found: {dict(model_counts)}")

if __name__ == "__main__":
    main()
                    
