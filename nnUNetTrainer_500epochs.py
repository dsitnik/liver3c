"""
Custom nnU-Net Trainer for 500 Epochs
This trainer overrides the default epoch count to train for exactly 500 epochs.

Usage:
    nnUNetv2_train DATASET_ID CONFIGURATION FOLD -tr nnUNetTrainer_500epochs

Example:
    nnUNetv2_train 100 3d_fullres 0 -tr nnUNetTrainer_500epochs
"""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_500epochs(nnUNetTrainer):
    """
    nnU-Net trainer configured for 500 epochs instead of the default 1000.

    This is useful when you want faster training cycles or have limited time/resources.
    All other training parameters remain the same as the base nnUNetTrainer.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: str = 'cuda'):
        """
        Initialize trainer with custom epoch count.

        Args:
            plans: Training plans dictionary
            configuration: Configuration name (e.g., '2d', '3d_fullres')
            fold: Fold number for cross-validation
            dataset_json: Dataset metadata
            unpack_dataset: Whether to unpack the dataset
            device: Device to use for training ('cuda' or 'cpu')
        """
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        # Override the default epoch settings
        # Set both attributes for compatibility with different nnU-Net versions
        self.num_epochs = 500
        self.max_num_epochs = 500
