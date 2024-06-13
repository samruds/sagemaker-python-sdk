# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

###
tox -e py310 -- tests/integ/sagemaker/serve/test_serve_tei.py

##


from __future__ import absolute_import

import pytest
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.serve.builder.model_builder import ModelBuilder, Mode

from tests.integ.sagemaker.serve.constants import (
    HF_DIR,
    PYTHON_VERSION_IS_NOT_310,
    SERVE_SAGEMAKER_ENDPOINT_TIMEOUT,
)

from tests.integ.timeout import timeout
from tests.integ.utils import cleanup_model_resources
import logging

^^^ Imports

logger = logging.getLogger(__name__)

sample_input = {"inputs": "What is Deep Learning?"}

loaded_response = []

^^^ sample inputs and outputs for schema builder


@pytest.fixture
def model_input():
    return {"inputs": "What is Deep Learning?"}


^^^ pytest.fixture is way to re-use implementation and plug it in across tests


@pytest.fixture
def model_builder_model_schema_builder():
    return ModelBuilder(
        model_path=HF_DIR,
        model="BAAI/bge-m3",
        schema_builder=SchemaBuilder(sample_input, loaded_response),
    )


@pytest.fixture
def model_builder(request):
    return request.getfixturevalue(request.param)


@pytest.mark.skipif(
    PYTHON_VERSION_IS_NOT_310,
    reason="Testing feature needs latest metadata",
)
@pytest.mark.parametrize("model_builder", ["model_builder_model_schema_builder"], indirect=True)
^^ patching object against fixture
def test_tei_sagemaker_endpoint(sagemaker_session, model_builder, model_input):
    logger.info("Running in SAGEMAKER_ENDPOINT mode...")

    ^^^ used for debugging

    caught_ex = None
    iam_client = sagemaker_session.boto_session.client("iam")
    role_arn = iam_client.get_role(RoleName="SageMakerRole")["Role"]["Arn"]

    ^^^ authorization role for integ test to invoke "real" sagemaker


    model = model_builder.build(
        mode=Mode.SAGEMAKER_ENDPOINT, role_arn=role_arn, sagemaker_session=sagemaker_session
    )

    ^^^ calling ModelBuilder in Endpoint mode, notice no Local mode here

    with timeout(minutes=SERVE_SAGEMAKER_ENDPOINT_TIMEOUT): << timeout to end test if it takes
    too long
        try:
            logger.info("Deploying and predicting in SAGEMAKER_ENDPOINT mode...")
            predictor = model.deploy(instance_type="ml.g5.2xlarge", initial_instance_count=1) <- \
                calling deploy to get a predictor
            predictor.predict(model_input) <- where we perform inference
            assert predictor is not None
        except Exception as e: <- catch any exceptions here
            caught_ex = e
        finally: < - clean up any resources , as these are actual prod resources
            cleanup_model_resources(
                sagemaker_session=model_builder.sagemaker_session,
                model_name=model.name,
                endpoint_name=model.endpoint_name,
            )
            if caught_ex:
                logger.exception(caught_ex)
                assert False, f"{caught_ex} was thrown when running tei sagemaker endpoint test"
