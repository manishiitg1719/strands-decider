# Deploying strands decider

You can deploy the `strands-decider` pip server (`strands-decider serve`,
[docs/inference.md, Serve](../docs/inference.md#serve)) as-is to any compute. Below is a list of
recommended ways to host this model yourself.

## Your machine

Run `strands-decider serve` on Linux with CUDA, on a Mac, or on CPU. See
[docs/inference.md, Environments for serving](../docs/inference.md#environments-for-serving).

## Amazon SageMaker

A real-time endpoint in your AWS account, on a GPU or CPU instance. One CDK stack copies the model
to a private S3 bucket, builds the image in AWS CodeBuild and creates the endpoint; `cdk destroy`
removes it. See
[sagemaker/README.md](sagemaker/README.md).

## Using it from an application

Once a server is running, anything that can send `POST /v1/systemone` can call it
([docs/inference.md, Serve](../docs/inference.md#serve)). The
[Strands agent example](../examples/strands/README.md) shows a complete application, and
[its hosted-endpoint section](../examples/strands/README.md#against-a-hosted-endpoint) runs it
against a SageMaker endpoint.
