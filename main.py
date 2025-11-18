import time
from batch_router.core.base import BatchConfig, BatchStatus, ProviderId
from batch_router.providers import GoogleGenAIProvider
from batch_router.core.input import InputBatch, InputRequest, InputMessage, InputMessageRole
from batch_router.core.base import TextContent, InferenceParams

provider = GoogleGenAIProvider(api_key="your-api-key")

requests = [
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
input_batch = InputBatch(
    requests=requests
)

google_genai_batch = input_batch.with_config(
    config=BatchConfig(
        provider_id=ProviderId.GOOGLE,
        model_id="gemini-2.5-flash"
    )
)

# You can also use the same input batch for multiple providers by configuring it with different models and providers.
# openai_batch = input_batch.with_config(
#     config=BatchConfig(
#         provider_id=ProviderId.OPENAI,
#         model_id="gpt-5-mini"
#     )
# )

google_genai_batch_id = provider.send_batch(google_genai_batch)

while provider.poll_status(google_genai_batch_id) != BatchStatus.COMPLETED:
    time.sleep(5)
    print(f"Batch {google_genai_batch_id} is {provider.poll_status(google_genai_batch_id)}")

google_genai_batch = provider.get_results(google_genai_batch_id)
for request in google_genai_batch.requests:
    print(f"Request {request.custom_id} has text {request.messages[0].contents[0].text}")
