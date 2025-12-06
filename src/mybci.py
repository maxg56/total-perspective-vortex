#!/usr/bin/env python3
"""
mybci.py - Main entry point for EEG Brain-Computer Interface

Usage:
    python mybci.py <subject> <run> <mode> [options]

Arguments:
    subject     Subject number (1-109)
    run         Run number (3-14, see below for task types)
    mode        'train' or 'predict'

Run types:
    3, 7, 11    - Motor execution (left fist vs right fist)
    4, 8, 12    - Motor imagery (left fist vs right fist)
    5, 9, 13    - Motor execution (both fists vs both feet)
    6, 10, 14   - Motor imagery (both fists vs both feet)

Examples:
    python mybci.py 4 14 train      # Train on subject 4, run 14
    python mybci.py 4 14 predict    # Predict on subject 4, run 14
    python mybci.py 1 6 train --pipeline csp_svm
    python mybci.py 1 6 train --compare
"""

import os
import sys
import argparse
import logging

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from constants import MIN_SUBJECT, MAX_SUBJECT, VALID_RUNS, TARGET_ACCURACY
from train import train_subject, compare_pipelines
from predict import run_prediction
from preprocess import preprocess_subject, get_run_type
from pipeline import list_pipelines

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='EEG Brain-Computer Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Run types:
    3, 7, 11    - Motor execution (left fist vs right fist)
    4, 8, 12    - Motor imagery (left fist vs right fist)
    5, 9, 13    - Motor execution (both fists vs both feet)
    6, 10, 14   - Motor imagery (both fists vs both feet)

Examples:
    python mybci.py 4 14 train
    python mybci.py 4 14 predict
    python mybci.py 1 6 train --pipeline csp_svm
    python mybci.py 1 6 train --compare
        """
    )

    parser.add_argument('subject', type=int,
                        help='Subject number (1-109)')
    parser.add_argument('run', type=int,
                        help='Run number (3-14)')
    parser.add_argument('mode', choices=['train', 'predict'],
                        help='Mode: train or predict')

    # Training options
    parser.add_argument('--pipeline', '-p', type=str, default='csp_lda',
                        choices=list_pipelines(),
                        help='Pipeline to use (default: csp_lda)')
    parser.add_argument('--cv', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare all pipelines')

    # Model options
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory for saved models (default: models)')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Specific model path to load (for predict mode)')

    # CSP options
    parser.add_argument('--n-components', type=int, default=6,
                        help='Number of CSP components (default: 6)')
    parser.add_argument('--reg', type=float, default=None,
                        help='CSP regularization (default: None)')

    # Verbose options
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Reduce output verbosity')

    # Visualization options
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable plot generation')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save plots to disk (plots/ directory)')

    return parser.parse_args()


def validate_args(args):
    """Validate command line arguments."""
    # Validate subject number
    if args.subject < MIN_SUBJECT or args.subject > MAX_SUBJECT:
        logger.error(f"Subject must be between {MIN_SUBJECT} and {MAX_SUBJECT}, got {args.subject}")
        print(f"Error: Subject must be between {MIN_SUBJECT} and {MAX_SUBJECT}, got {args.subject}")
        sys.exit(1)

    # Validate run number
    if args.run not in VALID_RUNS:
        logger.error(f"Run must be one of {VALID_RUNS}, got {args.run}")
        print(f"Error: Run must be one of {VALID_RUNS}, got {args.run}")
        sys.exit(1)


def print_header(args):
    """Print program header."""
    print("\n" + "=" * 60)
    print("    Total Perspective Vortex - EEG BCI System")
    print("=" * 60)
    print(f"Subject: {args.subject}")
    print(f"Run: {args.run} ({get_run_type(args.run)})")
    print(f"Mode: {args.mode}")
    print(f"Pipeline: {args.pipeline}")
    print("=" * 60)


def mode_train(args):
    """Execute training mode."""
    verbose = not args.quiet
    plot = not args.no_plot
    save_plots = args.save_plots

    if args.compare:
        # Compare all pipelines
        print("\nLoading and preprocessing data...")
        X, y, epochs = preprocess_subject(args.subject, [args.run])
        print(f"Data shape: {X.shape}")
        print(f"Labels: {len(y)} epochs")

        results = compare_pipelines(X, y, cv=args.cv, verbose=verbose,
                                   plot=plot, save_plots=save_plots)

        # Find best and train with it
        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            best_name = max(valid_results.items(), key=lambda x: x[1]['mean'])[0]
            print(f"\nTraining best pipeline: {best_name}")
            args.pipeline = best_name

    # Train model
    pipeline_kwargs = {}
    if 'csp' in args.pipeline:
        pipeline_kwargs['n_components'] = args.n_components
        if args.reg is not None:
            pipeline_kwargs['reg'] = args.reg

    pipeline, scores = train_subject(
        args.subject,
        [args.run],
        pipeline_name=args.pipeline,
        model_dir=args.model_dir,
        cv=args.cv,
        plot=plot,
        save_plots=save_plots,
        **pipeline_kwargs
    )

    # Print final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Cross-validation scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    if scores.mean() >= TARGET_ACCURACY:
        print(f"\n*** TARGET ACCURACY ({TARGET_ACCURACY:.0%}) ACHIEVED! ***")
    else:
        print(f"\n*** Target accuracy not reached (need {TARGET_ACCURACY - scores.mean():.4f} more) ***")

    return 0


def mode_predict(args):
    """Execute prediction mode."""
    verbose = not args.quiet

    # Run prediction
    results = run_prediction(
        args.subject,
        [args.run],
        model_path=args.model_path,
        model_dir=args.model_dir,
        pipeline_name=args.pipeline,
        verbose=verbose
    )

    # Print final summary
    print("\n" + "=" * 60)
    print("PREDICTION COMPLETE")
    print("=" * 60)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Average prediction time: {results['avg_time']*1000:.2f} ms")
    print(f"Max prediction time: {results['max_time']*1000:.2f} ms")
    print(f"Time limit (2s): {'PASSED' if results['within_time_limit'] else 'FAILED'}")

    if results['accuracy'] >= TARGET_ACCURACY:
        print(f"\n*** TARGET ACCURACY ({TARGET_ACCURACY:.0%}) ACHIEVED! ***")
    else:
        print(f"\n*** Target accuracy not reached ***")

    return 0


def main():
    """Main entry point."""
    args = parse_args()
    validate_args(args)
    print_header(args)

    try:
        if args.mode == 'train':
            return mode_train(args)
        elif args.mode == 'predict':
            return mode_predict(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        print("\nInterrupted by user")
        return 1
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\nError: {e}")
        print("Have you trained a model first? Run: python mybci.py <subject> <run> train")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
