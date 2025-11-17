import time
from batch_router.core.base.batch import BatchStatus
from batch_router.providers import vLLMProvider
from batch_router.core.input import InputBatch, InputRequest, InputMessage, InputMessageRole, InputRequestConfig
from batch_router.core.base import TextContent, InferenceParams

provider = vLLMProvider(model_path="./gemma_3_270m")

input_batch = InputBatch(
    requests=[
        InputRequest(
            custom_id=f"request_{i+1}",
            messages=[
                InputMessage(
                    role=InputMessageRole.USER,
                    contents=[
                        TextContent(text="Hello, tell me a story.")
                    ]
                )
            ],
            params=InferenceParams(
                max_output_tokens=100
            ),
            config=InputRequestConfig(
                model_id="./gemma_3_270m",
                provider_id=provider.provider_id
            )
        )
        for i in range(10)
    ]
)

response_batch_id = provider.send_batch(input_batch)

while provider.poll_status(response_batch_id) != BatchStatus.COMPLETED:
    time.sleep(3)
    print(f"Batch {response_batch_id} is {provider.poll_status(response_batch_id)}")

response_batch = provider.get_results(response_batch_id)
for request in response_batch.requests:
    print(f"Request {request.custom_id} has text {request.messages[0].contents[0].text}")
