import time
from batch_router.core.base import BatchConfig, BatchStatus, ProviderId
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
                        TextContent(text="Hello, how are you?")
                    ]
                )
            ],
            params=InferenceParams(
                max_output_tokens=100
            )
        )
        for i in range(10)
    ]
)

vllm_batch = input_batch.with_config(
    config=BatchConfig(
        provider_id=ProviderId.VLLM,
        model_id="./gemma_3_270m"
    )
)

vllm_batch_id = provider.send_batch(vllm_batch)

while provider.poll_status(vllm_batch_id) != BatchStatus.COMPLETED:
    time.sleep(3)
    print(f"Batch {vllm_batch_id} is {provider.poll_status(vllm_batch_id)}")

vllm_batch = provider.get_results(vllm_batch_id)
for request in vllm_batch.requests:
    print(f"Request {request.custom_id} has text {request.messages[0].contents[0].text}")
