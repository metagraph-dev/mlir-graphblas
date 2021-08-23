import os
import distutils.core
import subprocess
import json
import jinja2
import itertools
import random
import re
import pathlib
import time
import zlib

from typing import Any, Dict, List, Tuple, Generator, Callable, Literal, Union, Optional

##########
# Config #
##########


def pytest_configure(config: "_pytest.config.Config"):
    if not hasattr(config, "workerinput"):  # only run once on main process
        distutils.core.run_setup(
            "./setup.py", script_args=["build_ext", "--inplace"], stop_after="run"
        )
        # Ensure graphblas-opt is built
        subprocess.run(["python", os.path.join("mlir_graphblas", "src", "build.py")])

    return


def pytest_addoption(parser: "_pytest.config.argparsing.Parser"):
    parser.addoption(
        "--filecheck-sampling",
        action="store",
        default="default",
        help="Method to sample the space of the templatized FileCheck tests.",
    )
    return


def pytest_generate_tests(metafunc: "_pytest.python.Metafunc"):
    if metafunc.function.__name__ == "test_filecheck_mlir":
        parameterize_templatized_filecheck_test(metafunc)
    return


###########################################
# Parameterization of test_filecheck_mlir #
###########################################

NAMED_PARAMETER_VALUE_CHOICES = {
    "STANDARD_ELEMENT_TYPES": [
        "i1",
        "i4",
        "i8",
        "i16",
        "i32",
        "i64",
        "f16",
        "bf16",
        "f32",
        "f64",
        "f80",
        "f128",
    ],
    "MATRIX_APPLY_OPERATORS": ["min"],
    "SPARSITY_TYPES": ["dense", "compressed", "singleton"],
    "MATRIX_MULTIPLY_SEMIRINGS": ["plus_times", "plus_pair", "plus_plus"],
}

PREFILTER_FUNCTIONS = {None: lambda *args: True}

# Prefilter Functions


def prefilter_func(
    func: Callable[[Dict[str, str]], bool]
) -> Callable[[Dict[str, str]], bool]:
    if func.__name__ in PREFILTER_FUNCTIONS:
        raise Exception(
            f"{repr(func.__name__)} is already in use as prefilter function name."
        )
    PREFILTER_FUNCTIONS[func.__name__] = func
    return func


@prefilter_func
def different_thunk_and_element_type(parameter_dict: Dict[str, str]) -> bool:
    return parameter_dict["element_type"] != parameter_dict["thunk_type"]


@prefilter_func
def sparse_but_not_compressed_sparse(parameter_dict: Dict[str, str]) -> bool:
    sparsity0 = parameter_dict["sparsity0"]
    sparsity1 = parameter_dict["sparsity1"]
    return "compressed" in (sparsity0, sparsity1) and (
        (sparsity0, sparsity1) != ("dense", "compressed")
    )


# Template Expansion


def lazy_list_shuffler(input_list: list) -> Generator[Any, None, None]:
    """
    This generator yields an unseen item at random from the input_list.
    Upon each call to the generator, it'll randomly select an index
    until an unseen index is found.

    Running this generator to exhaustion is approximately O(n**2).
    This should only be used in cases where the generator is NOT expected
    to be exhausted and is expected to run a few times, in
    which case the expected running time is Theta(1).
    """
    seen_indices = set()
    for _ in range(len(input_list)):
        while (index := random.randint(0, len(input_list) - 1)) in seen_indices:
            pass
        seen_indices.add(index)
        yield input_list[index]
    return


def parameter_tuples_from_templates(
    sampling: Union[int, float, Literal["exhaustive", "bucket"]],
    seed: Optional[int] = None,
) -> List[Tuple[str, Tuple[str, str, Dict[str, str]]]]:
    """
    Returns a list of tuples of the form (
        "mlir_code",
        "test command jinja template",
        "/template/file/location",
        {"template_parameter_0": "template_parameter_value_0", ...}
    )

    If sampling is a float (in the range [0, 1]), the number of parameter
    tuples returned will be (total_num_possible_cases * sampling) for each
    template.

    If sampling is an int, the number of parameter tuples returned will
    be (sampling) for each template.

    If sampling is "exhaustive", we will return all possible parameter
    tuples for each template.

    If sampling is "bucket", we will yield parameter tuples as follows:
        For each template parameter name:
            Randomly sample one template parameter value for all other
                template parameter names
            Yield the pytest param for the set of template parameter
    """
    if seed is not None:
        random.seed(seed)
    parameter_tuples = []
    current_module_dir = pathlib.Path(__file__).parent.absolute()
    template_files = (
        os.path.join(root, f)
        for root, _, files in os.walk(current_module_dir, followlinks=True)
        for f in files
        if f.endswith(".template.mlir")
    )
    for template_file in template_files:
        # Parse the template file
        with open(template_file, "r") as f:
            json_sting, mlir_template = f.read().split("### START TEST ###")
        test_spec: dict = json.loads(json_sting)
        mlir_template = jinja2.Template(mlir_template, undefined=jinja2.StrictUndefined)
        if "parameters" not in test_spec:
            raise ValueError(
                f"{template_file} does not contain a valid test specification as "
                'it does not specify a value for the key "parameters".'
            )
        elif "run" not in test_spec or not isinstance(test_spec["run"], str):
            raise ValueError(
                f"{template_file} does not contain a valid test specification as "
                'it does not specify a valid value for the key "run".'
            )
        prefilter_name = test_spec.get("prefilter")
        parameter_dict_filter = PREFILTER_FUNCTIONS.get(prefilter_name)
        if parameter_dict_filter is None:
            raise NameError(f"Unknown prefilter function named {repr(prefilter_name)}.")

        # Grab test running command
        test_execution_command_template = test_spec["run"]

        # Grab parameter choices
        parameter_choices: Dict[str, List[str]] = dict()
        for parameter_name, parameter_value_choices in test_spec["parameters"].items():
            if (
                isinstance(parameter_value_choices, str)
                and parameter_value_choices in NAMED_PARAMETER_VALUE_CHOICES
            ):
                parameter_choices[parameter_name] = NAMED_PARAMETER_VALUE_CHOICES[
                    parameter_value_choices
                ]
            elif isinstance(parameter_value_choices, list):
                parameter_choices[parameter_name] = parameter_value_choices
            else:
                raise ValueError(
                    f"{repr(parameter_value_choices)} does not specify a valid set of parameter values."
                )

        # Handle each sampling case separately
        if sampling == "bucket":
            parameter_dicts = []
            for parameter_name, parameter_possible_values in parameter_choices.items():
                for parameter_value in parameter_possible_values:
                    # Set up lazy iterators to randomly grab the values of all the other parameters
                    other_parameter_names = [
                        name
                        for name in parameter_choices.keys()
                        if name != parameter_name
                    ]
                    other_parameter_choices_values = (
                        parameter_choices[name] for name in other_parameter_names
                    )
                    other_parameter_choices_values = map(
                        lazy_list_shuffler, other_parameter_choices_values
                    )
                    other_parameter_value_tuples = itertools.product(
                        *other_parameter_choices_values
                    )
                    other_parameter_dicts = (
                        dict(zip(other_parameter_names, other_parameter_values))
                        for other_parameter_values in other_parameter_value_tuples
                    )
                    # Go through possible parameter dicts until we find a valid one
                    for parameter_dict in other_parameter_dicts:
                        parameter_dict[parameter_name] = parameter_value
                        if parameter_dict_filter(parameter_dict):
                            parameter_tuples.append(
                                (
                                    generate_test_id_string(
                                        template_file, parameter_dict
                                    ),
                                    (
                                        mlir_template.render(**parameter_dict),
                                        test_execution_command_template,
                                        template_file,
                                        parameter_dict,
                                    ),
                                )
                            )
                            break
        else:
            if isinstance(sampling, int):

                def sampling_method(parameter_dicts):
                    parameter_dicts = list(parameter_dicts)
                    num_samples = min(sampling, len(parameter_dicts))
                    return random.sample(parameter_dicts, num_samples)

            elif isinstance(sampling, float):
                if sampling < 0 or sampling > 1:
                    raise ValueError(
                        f"Portion of parameter dicts to sample must be "
                        f"in the range [0, 1], got {sampling}."
                    )

                def sampling_method(parameter_dicts):
                    parameter_dicts = list(parameter_dicts)
                    num_samples = int(len(parameter_dicts) * sampling)
                    return random.sample(parameter_dicts, num_samples)

            elif sampling == "default":

                def sampling_method(parameter_dicts):
                    for parameter_dict in parameter_dicts:
                        return [parameter_dict]
                    return []

            elif sampling == "exhaustive":

                def sampling_method(parameter_dicts):
                    return parameter_dicts

            else:
                raise ValueError(
                    f"{repr(sampling)} is not a supported sampling method."
                )

            # Grab all possible parameter dicts
            parameter_names = parameter_choices.keys()
            parameter_value_tuples = itertools.product(*parameter_choices.values())
            all_parameter_dicts = (
                dict(zip(parameter_names, parameter_values))
                for parameter_values in parameter_value_tuples
            )

            # Find the requested parameter dicts
            parameter_dicts = filter(parameter_dict_filter, all_parameter_dicts)
            parameter_dicts = sampling_method(parameter_dicts)

            # Append one parameter dict for each test case
            for parameter_dict in parameter_dicts:
                parameter_tuples.append(
                    (
                        generate_test_id_string(template_file, parameter_dict),
                        (
                            mlir_template.render(**parameter_dict),
                            test_execution_command_template,
                            template_file,
                            parameter_dict,
                        ),
                    )
                )

    return parameter_tuples


def generate_test_id_string(template_file: str, parameter_dict: Dict[str, str]) -> str:
    return "".join(c for c in template_file if c.isalnum()) + "".join(
        f"({re.escape(key)}:{re.escape(parameter_dict[key])})"
        for key in sorted(parameter_dict.keys())
    )


def parameterize_templatized_filecheck_test(metafunc: "_pytest.python.Metafunc"):
    sampling_method_string = metafunc.config.getoption("--filecheck-sampling")
    if sampling_method_string.isdigit():
        sampling = int(sampling_method_string)
    elif re.match("^\d+(\.\d+)?$", sampling_method_string):
        sampling = float(sampling_method_string)
    else:
        sampling = sampling_method_string

    # All the workers need to have run the exact same tests. Thus, they need to use
    # the exact same random seed when sampling the tests (all the workers redundantly
    # do this; not sure how to avoid this redundant work). I do this by making the
    # testrunuid the seed (via all the zlib.adler32 stuff) if the worker is running
    # this code. Since the main/non-worker process doesn't run any tests (so nothing
    # bad happens if we randomly sample different tests) and since the main process's
    # metafunc.config doesn't have a workerinput attribute, we just use time.time().
    seed = (
        zlib.adler32(metafunc.config.workerinput["testrunuid"].encode())
        if hasattr(metafunc.config, "workerinput")
        else time.time()
    )
    ids, parameter_values = zip(*parameter_tuples_from_templates(sampling, seed))
    metafunc.parametrize(
        ["mlir_code", "test_command_template", "template_file", "parameter_dict"],
        parameter_values,
        ids=ids,
    )

    # Ensure graphblas-opt is built
    subprocess.run(["python", os.path.join("mlir_graphblas", "src", "build.py")])

    return
