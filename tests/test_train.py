import os

import pytest

from ecodse_funtime_alpha.train import get_args


class TestArgparse(object):

    def test_argparsenormal(self):
        fakearg = ['--imagepath=./', '--labelpath=fakedir/name.csv',
                   '--seed=1', '--kernels=10', '--ksize=1',
                   '--lr=0.01', '--nepoch=2', '--batchsize=4'
                   ]
        args = get_args(fakearg)
        assert args.imagepath == './'
        assert args.labelpath == 'fakedir/name.csv'
        assert args.seed == 1
        assert args.kernels == 10
        assert args.ksize == 1
        assert args.lr == 0.01
        assert args.nepoch == 2
        assert args.batchsize == 4

    @pytest.mark.xfail(raises=SystemExit)
    def test_argparse_lr(self):
        fakearg = ['--lr=a']
        _ = get_args(fakearg)

    @pytest.mark.xfail(raises=SystemExit)
    def test_argparse_seed(self):
        fakearg = ['--seed=a']
        _ = get_args(fakearg)

    def test_argparse_imagepath(self):
        fakearg = ['--imagepath=notavalidpath']
        args = get_args(fakearg)
        assert not os.path.isdir(args.imagepath)

    def test_argparse_labelpath(self):
        fakearg = ['--labelpath=invalid.csv']
        args = get_args(fakearg)
        assert not os.path.exists(args.labelpath)
