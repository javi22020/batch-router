# Batch Inference Mistral API Guide

Batching allows you to run inference on large inputs in parallel, reducing costs while running large workloads.

Prepare batch file

# Copy Prepare batch file

### Prepare and Upload your Batch File

A batch is composed of a list of API requests. The structure of an individual request includes:

*   A unique `custom_id` for identifying each request and referencing results after completion
*   A `body` object with message information

Here's an example of how to structure a batch request:

```
{"custom_id": "0", "body": {"max_tokens": 100, "messages": [{"role": "user", "content": "What is the best French cheese?"}]}}
{"custom_id": "1", "body": {"max_tokens": 100, "messages": [{"role": "user", "content": "What is the best French wine?"}]}}
```

```
{"custom_id": "0", "body": {"max_tokens": 100, "messages": [{"role": "user", "content": "What is the best French cheese?"}]}}
{"custom_id": "1", "body": {"max_tokens": 100, "messages": [{"role": "user", "content": "What is the best French wine?"}]}}
```

A batch `body` object can be any **valid request body for the endpoint** you are using. Below are examples of batch files for different endpoints, they have their `body` match the endpoint's request body.

Chat Completion

Structured Outputs

Embeddings

OCR

Transcriptions

![Image 1: Cat head](https://docs.mistral.ai/_next/image?url=%2Fassets%2Fsprites%2Fcat_head.png&w=48&q=75) ¡Meow! Click one of the tabs above to learn more.

Save your batch into a .jsonl file. Once saved, you can upload your batch input file to ensure it is correctly referenced when initiating batch processes.

There are 2 main ways of uploading and running a batch:

**A.** Via AI Studio ( Recommended ):

*   Upload your files here: [https://console.mistral.ai/build/files](https://console.mistral.ai/build/files)
    
    *   Upload the file in the format described previously.
    *   Set `purpose` to Batch Processing.
*   Start and Manage your batches here: [https://console.mistral.ai/build/batches](https://console.mistral.ai/build/batches)
    
    *   Create and start a job by providing the `files`, `endpoint` and `model`. You wont need to use the API to upload your files and/or create batching jobs.

**B.** Via the API, explained below.

To upload your batch file, you need to use the `files` endpoint.

python

```
from mistralai import Mistral
import os

api_key = os.environ["MISTRAL_API_KEY"]

client = Mistral(api_key=api_key)

batch_data = client.files.upload(
    file={
        "file_name": "test.jsonl",
        "content": open("test.jsonl", "rb")
    },
    purpose = "batch"
)
```

```
from mistralai import Mistral
import os

api_key = os.environ["MISTRAL_API_KEY"]

client = Mistral(api_key=api_key)

batch_data = client.files.upload(
    file={
        "file_name": "test.jsonl",
        "content": open("test.jsonl", "rb")
    },
    purpose = "batch"
)
```

Batch Creation

# Copy Batch Creation

### Create a new Batch Job

Create a new batch job, it will be queued for processing.

*   `input_files`: a list of the batch input file IDs.
*   `model`: you can only use one model (e.g., `codestral-latest`) per batch. However, you can run multiple batches on the same files with different models if you want to compare outputs.
*   `endpoint`: we currently support `/v1/embeddings`, `/v1/chat/completions`, `/v1/fim/completions`, `/v1/moderations`, `/v1/chat/moderations`, `/v1/ocr`, `/v1/classifications`, `/v1/conversations`, `/v1/audio/transcriptions`.
*   `metadata`: optional custom metadata for the batch.

python

```
created_job = client.batch.jobs.create(
    input_files=[batch_data.id],
    model="mistral-small-latest",
    endpoint="/v1/chat/completions",
    metadata={"job_type": "testing"}
)
```

```
created_job = client.batch.jobs.create(
    input_files=[batch_data.id],
    model="mistral-small-latest",
    endpoint="/v1/chat/completions",
    metadata={"job_type": "testing"}
)
```

Get/Retrieve

# Copy Get/Retrieve

### Retrieve your Batch Job

Once batch sent, you will want to retrieve a lot of information such as:

*   The status of the batch job
*   The results of the batch job
*   The list of batch jobs

Get a batch job details

## Copy Get a batch job details

You can retrieve the details of a batch job by its ID.

python

`retrieved_job = client.batch.jobs.get(job_id=created_job.id)`

`retrieved_job = client.batch.jobs.get(job_id=created_job.id)`

Get batch job results

## Copy Get batch job results

Once the batch job is completed, you can easily download the results.

python

```
output_file_stream = client.files.download(file_id=retrieved_job.output_file)

# Write and save the file
with open('batch_results.jsonl', 'wb') as f:
    f.write(output_file_stream.read())
```

```
output_file_stream = client.files.download(file_id=retrieved_job.output_file)

# Write and save the file
with open('batch_results.jsonl', 'wb') as f:
    f.write(output_file_stream.read())
```

List batch jobs

## Copy List batch jobs

You can view a list of your batch jobs and filter them by various criteria, including:

*   Status: `QUEUED`, `RUNNING`, `SUCCESS`, `FAILED`, `TIMEOUT_EXCEEDED`, `CANCELLATION_REQUESTED` and `CANCELLED`
*   Metadata: custom metadata key and value for the batch

python

```
list_job = client.batch.jobs.list(
    status="RUNNING",
    metadata={"job_type": "testing"}
)
```

```
list_job = client.batch.jobs.list(
    status="RUNNING",
    metadata={"job_type": "testing"}
)
```

Request Cancellation

# Copy Request Cancellation

### Cancel any Job

If you want to cancel a batch job, you can do so by sending a cancellation request.

python

`canceled_job = client.batch.jobs.cancel(job_id=created_job.id)`

`canceled_job = client.batch.jobs.cancel(job_id=created_job.id)`

An end-to-end example

# Copy An end-to-end example

Below is an end-to-end example of how to use the batch API from start to finish.

End-to-End Example

![Image 2: Cat head](https://docs.mistral.ai/_next/image?url=%2Fassets%2Fsprites%2Fcat_head.png&w=48&q=75) ¡Meow! Click one of the tabs above to learn more.

#### Contents

*   [Prepare batch file](https://docs.mistral.ai/capabilities/batch#prepare-batch-file)
*   [Batch Creation](https://docs.mistral.ai/capabilities/batch#batch-creation)
*   [Get/Retrieve](https://docs.mistral.ai/capabilities/batch#get-retrieve)
*   [Get a batch job details](https://docs.mistral.ai/capabilities/batch#get-a-batch-job-details)
*   [Get batch job results](https://docs.mistral.ai/capabilities/batch#get-batch-job-results)
*   [List batch jobs](https://docs.mistral.ai/capabilities/batch#list-batch-jobs)
*   [Request Cancellation](https://docs.mistral.ai/capabilities/batch#request-the-cancellation)
*   [An end-to-end example](https://docs.mistral.ai/capabilities/batch#an-end-to-end-example)
*   [FAQ](https://docs.mistral.ai/capabilities/batch#faq)
