from nilearn.image.image import mean_img
from nilearn.plotting import plot_epi, plot_roi
from nilearn.masking import compute_epi_mask
from glob import glob
from pathlib import Path
import os.path as op

# create a directory for figures
Path('figures').mkdir(exist_ok=True)


def basic_plots():
    """
    Plot and save a mean EPI image and an EPI mask
    :return:
    """
    for input in glob('input/sub*'):
        sub = op.basename(input)
        func = input + '/func/{}_task-oneback_run-01_bold.nii.gz'.format(sub)
        mean = mean_img(func)
        plot_epi(mean).savefig('figures/{}_mean-epi.png'.format(sub))
        mask_img = compute_epi_mask(func)
        mask_img.to_filename('{}_brain-mask.nii.gz'.format(sub))
        plot_roi(mask_img, mean).savefig('figures/{}_brainmask.png'.format(sub))


if __name__ == '__main__':
    basic_plots()