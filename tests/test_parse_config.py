import os

import pytest
import torch

import parse_config
from parse_config import ConfigParser
from utils import dict2obj


class TestParseConfig:

    def setup_class(self, ):
        args = {}

        def parse_args():
            return {}

        args['parse_args'] = parse_args
        self.args = dict2obj(args)

    def setup_mocker(self, mocker):
        mocker.patch.object(ConfigParser, "save_dir", return_value=dict2obj({"mkdir": lambda a: a}))
        mocker.patch.object(ConfigParser, "log_dir", return_value=dict2obj({"mkdir": lambda a: a}))
        mocker.patch.object(parse_config, "write_json")
        mocker.patch.object(parse_config, "setup_logging")

    def test_from_args_empty_config(self, mocker):
        with pytest.raises(AssertionError) as excinfo:
            args = self.args
            mocker.patch.object(args, "parse_args",
                                return_value=dict2obj({"device": None, "resume": None, "config": None}))
            config = ConfigParser.from_args(args)
        assert "Configuration file need to be specified. Add '-c config.json', for example." in str(excinfo.value)

    def test_from_args_load_resume_config(self, mocker):
        args = self.args
        self.setup_mocker(mocker)
        with mocker.patch.object(args, "parse_args", return_value=dict2obj(
                {"device": None, "resume": "tests/config.json", "config": None})):
            config = ConfigParser.from_args(args)
            assert config.config['name'] == 'Mnist_LeNet_Resume'
            args.parse_args.assert_called_once()

        with mocker.patch.object(args, "parse_args", return_value=dict2obj(
                {"device": None, "resume": "tests/config.json", "config": "./config.json"})):
            config = ConfigParser.from_args(args)
            # If both resume and config parameter set, the config will override the configs in resume one.
            assert config.config['name'] == 'Mnist_LeNet'
            args.parse_args.assert_called_once()

        with mocker.patch.object(args, "parse_args", return_value=dict2obj(
                {"device": "1,2", "resume": None, "config": "tests/config.json"})):
            config = ConfigParser.from_args(args)
            # If both resume and config parameter set, the config will override the configs in resume one.
            assert config.config['name'] == 'Mnist_LeNet_Resume'
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,2"
            args.parse_args.assert_called_once()

        assert ConfigParser.save_dir.mkdir.call_count == 3
        assert ConfigParser.log_dir.mkdir.call_count == 3

    def test_init_obj_without_args(self, mocker):
        args = self.args
        self.setup_mocker(mocker)
        with mocker.patch.object(args, "parse_args", return_value=dict2obj(
                {"device": None, "resume": "tests/config.json", "config": None})):
            config = ConfigParser.from_args(args)
            config.config['tests'] = {}
            config.config['tests']['type'] = "FakeClass"
            obj = config.init_obj('tests')
            assert obj.test_func()
            assert obj.test_func_with_param1() is None

    def test_init_obj_with_args(self, mocker):
        args = self.args
        self.setup_mocker(mocker)
        with mocker.patch.object(args, "parse_args", return_value=dict2obj(
                {"device": None, "resume": "tests/config.json", "config": None})):
            config = ConfigParser.from_args(args)
            config.config['tests'] = {}
            config.config['tests']['type'] = "FakeClass"
            config.config['tests']['args'] = {"param1": "content"}

            obj = config.init_obj('tests')
            assert obj.test_func()
            assert obj.test_func_with_param1() == "content"

    def test_init_obj_with_lib(self, mocker):
        args = self.args
        self.setup_mocker(mocker)

        mocker.patch.object(torch.optim, "SGD")
        with mocker.patch.object(args, "parse_args", return_value=dict2obj(
                {"device": None, "resume": "tests/config.json", "config": None})):
            config = ConfigParser.from_args(args)
            config.config['optimizer'] = {}
            config.config['optimizer']['type'] = "SGD"
            obj = config.init_obj('optimizer', torch.optim, {})
            obj()
            assert obj.call_count == 1

    def test_init_ftn_with_arg(self, mocker):
        args = self.args
        self.setup_mocker(mocker)
        with mocker.patch.object(args, "parse_args", return_value=dict2obj(
                {"device": None, "resume": "tests/config.json", "config": None})):
            config = ConfigParser.from_args(args)
            config.config['tests'] = ["fake_func1", "fake_func2"]

            ftns = config.init_ftn("tests", param1="param1")

            for ftn in ftns:
                assert ftn() == "param1"
