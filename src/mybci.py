#!/usr/bin/env python3
"""
mybci.py - Main entry point for EEG Brain-Computer Interface.

Usage
-----
    python mybci.py <subject> <run> <mode> [options]

Arguments
---------
    subject     Subject number (1-109)
    run         Run number (3-14, see below for task types)
    mode        'train' or 'predict'

Run types
---------
    3, 7, 11    - Motor execution (left fist vs right fist)
    4, 8, 12    - Motor imagery (left fist vs right fist)
    5, 9, 13    - Motor execution (both fists vs both feet)
    6, 10, 14   - Motor imagery (both fists vs both feet)

Examples
--------
    python mybci.py 4 14 train      # Train on subject 4, run 14
    python mybci.py 4 14 predict    # Predict on subject 4, run 14
    python mybci.py 1 6 train --pipeline csp_svm
    python mybci.py 1 6 train --compare
"""

import os
import sys
import argparse
import logging
import traceback

# Add src directory to path for imports and set it as working directory so that
# relative paths (models/, plots/, data/) resolve correctly regardless of where
# the script is invoked from.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC_DIR)
os.chdir(_SRC_DIR)

from constants import (MIN_SUBJECT, MAX_SUBJECT, VALID_RUNS,  # noqa: E402
                       EXPERIMENT_TARGETS, RUN_TO_EXPERIMENT, TARGET_ACCURACY)
from training import train_subject, compare_pipelines  # noqa: E402
from predict import run_prediction  # noqa: E402
from preprocess import preprocess_subject, get_run_type  # noqa: E402
from pipeline import list_pipelines  # noqa: E402
import display  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
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
    parser.add_argument('--show-plots', action='store_true',
                        help='Show plots interactively in a window')

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    if args.subject < MIN_SUBJECT or args.subject > MAX_SUBJECT:
        msg = f"Subject must be between {MIN_SUBJECT} and {MAX_SUBJECT}, got {args.subject}"
        logger.error(msg)
        print(f"Error: {msg}")
        sys.exit(1)

    if args.run not in VALID_RUNS:
        msg = f"Run must be one of {VALID_RUNS}, got {args.run}"
        logger.error(msg)
        print(f"Error: {msg}")
        sys.exit(1)


def mode_train(args: argparse.Namespace) -> int:
    """Execute training mode."""
    verbose = not args.quiet
    plot = not args.no_plot
    save_plots = args.save_plots
    show_plots = args.show_plots

    if args.compare:
        print("\nLoading and preprocessing data...")
        X, y, epochs = preprocess_subject(args.subject, [args.run])
        display.print_data_info(X, y)

        results = compare_pipelines(
            X, y, cv=args.cv, verbose=verbose,
            plot=plot, save_plots=save_plots, show_plots=show_plots)

        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            best_name = max(valid_results.items(), key=lambda x: x[1]['mean'])[0]
            print(f"\nTraining best pipeline: {best_name}")
            args.pipeline = best_name

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
        show_plots=show_plots,
        **pipeline_kwargs
    )

    exp_idx = RUN_TO_EXPERIMENT.get(args.run)
    exp_target = (
        EXPERIMENT_TARGETS.get(exp_idx, TARGET_ACCURACY) if exp_idx is not None else TARGET_ACCURACY
    )
    display.print_training_summary(scores, exp_idx, exp_target)

    return 0


def mode_predict(args: argparse.Namespace) -> int:
    """Execute prediction mode."""
    results = run_prediction(
        args.subject,
        [args.run],
        model_path=args.model_path,
        model_dir=args.model_dir,
        pipeline_name=args.pipeline,
        verbose=not args.quiet
    )

    exp_idx = RUN_TO_EXPERIMENT.get(args.run)
    exp_target = (
        EXPERIMENT_TARGETS.get(exp_idx, TARGET_ACCURACY) if exp_idx is not None else TARGET_ACCURACY
    )
    display.print_prediction_summary(results, exp_idx, exp_target)

    return 0


def main() -> int:
    """Execute main program logic."""
    args = parse_args()
    validate_args(args)
    display.print_header(args.subject, args.run, args.mode, args.pipeline, get_run_type(args.run))

    try:
        if args.mode == 'train':
            return mode_train(args)
        elif args.mode == 'predict':
            return mode_predict(args)
        return 0
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
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
